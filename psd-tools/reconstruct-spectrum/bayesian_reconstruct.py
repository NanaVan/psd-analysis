#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np
from scipy.ndimage import label, binary_dilation, median_filter

def estimate_bayesian_amplitude_v6(single_res, template, correlation_threshold=0.6):
    """
    融合版：局部去趋势 + 物理尺度投影 + 高相关收缩
    """
    if len(single_res) < 3:
        return 0.0
        
    # 1. 局部去趋势以提取形状置信度
    y_detrend = single_res - np.mean(single_res)
    t_detrend = template - np.mean(template)
    
    norm_y = np.linalg.norm(y_detrend)
    norm_t = np.linalg.norm(t_detrend)
    
    if norm_y == 0 or norm_t == 0:
        return 0.0
        
    correlation = np.dot(y_detrend, t_detrend) / (norm_y * norm_t)
    
    # 2. 形状匹配门槛
    if correlation < correlation_threshold:
        return 0.0
    
    # 3. 物理尺度投影 (MLE)
    alpha_ml = np.dot(single_res, template) / (np.dot(template, template) + 1e-9)
    
    # 4. 贝叶斯软收缩：使用 correlation 的高次幂压制不确定信号
    alpha = max(alpha_ml, 0) * (correlation ** 4)
    
    return alpha

def reconstruct_ion_psd(single_log_psd, avg_log_psd, baseline, k_prior=3.0, k_local=5.0):
    """
    双轨重构算法：
    - Track 1: 平均谱先验 (解决基线陷阱)
    - Track 2: 局部显著性 (捞出被掩盖的单帧尖峰)
    """
    # A. 提取平均谱先验 Mask
    signal_template = np.maximum(avg_log_psd - baseline, 0)
    std_bg = np.std(signal_template)
    prior_mask = signal_template > (np.median(signal_template) + k_prior * std_bg)
    prior_mask = binary_dilation(prior_mask, iterations=15)
    
    # B. 单帧局部显著性探测 (捞出平均谱里没有的峰)
    single_smooth = median_filter(single_log_psd, size=3)
    single_res = single_smooth - baseline
    
    # 计算局部信噪比：如果某点远高于周围中值，判定为潜在信号
    local_median = median_filter(single_res, size=50)
    local_std = np.std(single_res - local_median)
    saliency_mask = single_res > (local_median + k_local * local_std)
    
    # 合并 Mask：已知区域 + 突发区域
    combined_mask = prior_mask | saliency_mask
    
    # C. 分区域执行贝叶斯重构
    reconstructed_signal = np.zeros_like(single_log_psd)
    labels, n = label(combined_mask)
    
    for i in range(1, n + 1):
        region = (labels == i)
        y_local = single_res[region]
        
        # 确定模板：如果在先验区，用平均谱形状；如果是突发区，用理想高斯/单点形状
        if np.any(prior_mask[region]):
            t_local = signal_template[region]
        else:
            # 动态生成简易高斯模板 (假设掩盖的峰是窄峰)
            t_local = np.exp(-np.linspace(-2, 2, len(y_local))**2)
            
        alpha = estimate_bayesian_amplitude_v6(y_local, t_local, correlation_threshold=0.6)
        
        if alpha > 0:
            reconstructed_signal[region] = alpha * t_local
                
    return baseline + reconstructed_signal
