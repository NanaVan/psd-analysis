#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np
from scipy.ndimage import label, binary_dilation, median_filter

def estimate_bayesian_amplitude(single_psd_log, mean_rebuilt, baseline, 
                                 corr_threshold=0.65, 
                                 min_width=5,
                                 noise_floor_std=2.0):
    """
    基于平均谱形状先验的增强型单帧重建算法
    
    参数:
    - single_psd_log: 单帧数据的Log值 (np.log(data['psd_arrays'][i]))
    - mean_rebuilt:  平均谱重建后的纯信号结果 (作为先验形状)
    - baseline:      从平均谱中提取的基线 (作为单帧的参考底)
    - corr_threshold: 相关性门槛，压制"杂草"的关键
    - min_width:      最小峰宽限制 (物理稀疏性约束)
    - noise_floor_std: 用于估计局部置信度的噪声标准差倍数
    """
    
    # 1. 预处理：获取单帧的偏差信号 (Residual)
    # 注意：单帧由于涨落，log后的基准可能与平均谱基线有偏移
    residual = single_psd_log - baseline
    
    # 2. 识别平均谱中的"先验区域" (Prior Regions)
    # 找到平均谱中所有显著的峰作为模板
    peaks, props = find_peaks(mean_rebuilt, height=np.std(mean_rebuilt)*0.5)
    
    reconstructed_single = np.zeros_like(residual)
    
    # 3. 逐个模板匹配与振幅估计
    for peak_idx, center in enumerate(peaks):
        # 确定该峰的范围 (左边界和右边界)
        # 这里使用简单拓扑，也可以用 props['left_ips'] 等更精确的宽度
        width_half = 20 # 假设一个经验宽度，或根据 props 计算
        left = max(0, center - width_half)
        right = min(len(residual), center + width_half)
        
        # 提取模板形状和单帧观测段
        template_shape = mean_rebuilt[left:right]
        obs_segment = residual[left:right]
        
        # 只有当模板本身有意义时才继续
        if np.max(template_shape) <= 0:
            continue

        # 计算皮尔逊相关系数：验证形状吻合度
        norm_template = (template_shape - np.mean(template_shape)) / (np.std(template_shape) + 1e-9)
        norm_obs = (obs_segment - np.mean(obs_segment)) / (np.std(obs_segment) + 1e-9)
        corr = np.mean(norm_template * norm_obs)
        
        # 策略 A: 形状不匹配 -> 视为噪声，直接丢弃 (收紧判定门槛)
        if corr < corr_threshold:
            continue
            
        # 策略 B: 宽度过滤 -> 剔除孤立尖刺
        if (right - left) < min_width:
            continue

        # 4. 贝叶斯振幅估计 (Bayesian Amplitude Estimation)
        # 模型: obs = alpha * template + noise
        # 极大似然估计 alpha = dot(obs, template) / dot(template, template)
        # 考虑到只有正向信号，限制 alpha > 0
        num = np.sum(obs_segment * template_shape)
        den = np.sum(template_shape**2)
        alpha = max(0, num / (den + 1e-9))
        
        # 5. 局部能量门槛 (进一步压制杂草)
        # 如果估计出的信号峰值还不如局部噪声涨落显著，则收缩
        local_noise_estimate = np.std(obs_segment - alpha * template_shape)
        if alpha * np.max(template_shape) < noise_floor_std * local_noise_estimate:
            # 贝叶斯收缩 (Shrinkage)
            alpha *= (corr ** 2) # 相关性越低，压制越狠
            
        # 写入重建结果
        reconstructed_single[left:right] = np.maximum(reconstructed_single[left:right], alpha * template_shape)

    # 6. 全局正向约束与平滑
    reconstructed_single = np.maximum(reconstructed_single, 0)
    
    return reconstructed_single

# --- 使用示例 ---
# rebuilt_spectra = np.zeros_like(data['psd_arrays'])
# for i, _psd in enumerate(data['psd_arrays']):
#     single_log = np.log(_psd)
#     rebuilt_spectra[i,:] = reconstruct_with_shape_prior(
#         single_log, 
#         avg_log_psd,  # 这是你“先平均再log”得到的橙线
#         baseline,
#         k_prior=2.0   # 这个参数决定了平均谱中多小的峰会被纳入先验
#     )
