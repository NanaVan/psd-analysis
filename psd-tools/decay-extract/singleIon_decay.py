#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft

def analyze_ion(iq_data, fs=5e6, f_center=310e6, f_target=309.86222e6, t_start_ms=500, t_end_ms=800, f_sweep_range=800, decay=True):
    """
    v29: 动态频率追踪 + PyWavelets 脊线功率提取
    解决 f_target 在 ±300Hz 内随时间变动的问题
    decay:  True for detecting decayed single ion, False for detecting produced single ion
    """
    # 1. 精确切片
    n_start = int(t_start_ms * 1e-3 * fs)
    n_end = int(t_end_ms * 1e-3 * fs)
    data_slice = iq_data[n_start:n_end]
    
    # 2. 动态频率追踪 (使用短时高分辨率谱获取频率脊线)
    # nperseg 取 50000 约对应 1ms 的时间分辨率，足以跟踪数百 Hz 的慢漂移
    f, t_spec, Zxx = stft(data_slice, fs, window=('kaiser', 14), nperseg=50000, noverlap=49000, return_onesided=False)
    f_abs = f + f_center
    
    # 在目标范围内追踪每一时刻的最大功率频率
    mask = (f_abs > f_target - f_sweep_range/2) & (f_abs < f_target + f_sweep_range/2)
    f_track = []
    p_track = []
    
    for i in range(len(t_spec)):
        power_in_window = np.abs(Zxx[mask, i])
        max_idx = np.argmax(power_in_window)
        f_track.append(f_abs[mask][max_idx])
        p_track.append(power_in_window[max_idx]) # 顺便记录瞬时峰值功率
        
    f_track = np.array(f_track)
    avg_f_actual = np.mean(f_track)
    
    print(f"Frequency Tracking Active: {f_target/1e6:.6f} MHz ± 300Hz")
    print(f"Mean Tracked Freq: {avg_f_actual/1e6:.6f} MHz")

    # 3. 动态小波功率提取 (取追踪到的中位数频率作为主尺度，或利用追踪结果)
    # 为了体现随动，我们直接利用追踪到的脊线强度进行 CUSUM
    # stft 的 Zxx 本身就是某种意义上的宽窗小波，直接用它的脊线值最鲁棒
    power_db = 20 * np.log10(np.array(p_track) + 1e-15)
    t_plot = t_spec + (t_start_ms / 1000.0)
    
    # 4. CUSUM 判定
    # 这里的关键：如果信号消失，脊线功率会跌落到本底噪声的水平
    threshold = np.mean(power_db)
    diff = power_db - threshold
    if decay:
        cusum = np.cumsum(diff)
        event_idx = np.argmax(cusum)
    else:
        cusum = np.cumsum(diff)
        event_idx = np.argmin(cusum)

    event_time = t_plot[event_idx]

    # 5. 绘图
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
    
    # 频率追踪图
    ax1.plot(t_plot * 1000, (f_track - f_target), color='green')
    ax1.set_title("Frequency Drift Tracking (f_actual - f_target)")
    ax1.set_ylabel("Drift (Hz)")
    ax1.grid(True)

    # 脊线功率图
    ax2.plot(t_plot * 1000, power_db, label='Ridge Power (dB)')
    if decay:
        ax2.axvline(event_time * 1000, color='r', linestyle='--', label=f'Decay @ {event_time*1000:.2f}ms')
    else:
        ax2.axvline(event_time * 1000, color='r', linestyle='--', label=f'Produce @ {event_time*1000:.2f}ms')
    ax2.set_ylabel("Power (dB)")
    ax2.set_title("Instantaneous Power along the Frequency Ridge")
    ax2.legend()
    ax2.grid(True)

    # CUSUM
    ax3.plot(t_plot * 1000, cusum, color='brown')
    ax3.set_ylabel("CUSUM")
    ax3.set_xlabel("Time (ms)")
    ax3.set_title("CUSUM of Ridge Power")
    ax3.grid(True)

    plt.tight_layout()
    plt.show()

    return event_time


def analyze_ion_joint(iq_data, fs=5e6, f_center=310e6, 
                      f_m_target=309.86222e6, 
                      f_g_target=309.8639e6, 
                      t_start_ms=500, t_end_ms=800, 
                      f_sweep_range=800):
    """
    v32: 联合差分分析 (Joint Differential Analysis)
    思路 2: 结合激发态消失和基态出现的双重信息，提升定位精度并给出误差估算。
    """
    # 1. 数据切片
    n_start = int(t_start_ms * 1e-3 * fs)
    n_end = int(t_end_ms * 1e-3 * fs)
    data_slice = iq_data[n_start:n_end]
    
    # 2. STFT 基础分析 (采用 0.2ms 步进)
    # nperseg=50000 (10ms), noverlap=49000 (9.8ms step 0.2ms)
    f, t_spec, Zxx = stft(data_slice, fs, window=('kaiser', 14), nperseg=50000, noverlap=49000, return_onesided=False)
    f_abs = f + f_center
    t_plot = (t_spec + t_start_ms / 1000.0) * 1000  # 转换为 ms
    
    def track_ridge(target_freq):
        mask = (f_abs > target_freq - f_sweep_range/2) & (f_abs < target_freq + f_sweep_range/2)
        p_track = []
        for i in range(len(t_spec)):
            power_in_window = np.abs(Zxx[mask, i])
            p_track.append(np.max(power_in_window))
        return 20 * np.log10(np.array(p_track) + 1e-15)

    # 3. 分别追踪两条线的功率
    p_m_db = track_ridge(f_m_target) # 激发态 (应该是降)
    p_g_db = track_ridge(f_g_target) # 基态 (应该是升)
    
    # 4. 归一化并计算差分功率 (Differential Power)
    # 归一化目的是消除两条线本身的绝对强度差异
    p_m_norm = (p_m_db - np.mean(p_m_db)) / np.std(p_m_db)
    p_g_norm = (p_g_db - np.mean(p_g_db)) / np.std(p_g_db) if np.std(p_g_db)>0 else p_g_db
    
    # 构造综合跳变指标：S = P_g - P_m
    # 衰变发生时，P_g 变大，P_m 变小，S 会发生剧烈的正向跳变
    diff_signal = p_g_norm - p_m_norm
    
    # 5. CUSUM 联合判定
    # 寻找 diff_signal 跨越均值的转折点
    cusum_joint = np.cumsum(diff_signal - np.mean(diff_signal))
    # 对于低->高的跳变，转折点在 CUSUM 的最小值处 (谷底)
    event_idx = np.argmin(cusum_joint)
    event_time_ms = t_plot[event_idx]

    # 6. 误差估算
    # 计算单独判定时的差异作为物理不确定度
    idx_m = np.argmax(np.cumsum(p_m_db - np.mean(p_m_db))) # 单独看消失
    idx_g = np.argmin(np.cumsum(p_g_db - np.mean(p_g_db))) # 单独看出现
    time_diff = abs(t_plot[idx_m] - t_plot[idx_g])
    
    # 综合不确定度 = sqrt( 算法步进^2 + (物理偏差/2)^2 )
    step_precision = (50000 - 49000) / fs * 1000 # 0.2ms
    uncertainty = np.sqrt(step_precision**2 + (time_diff/2)**2)

    # 7. 绘图
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 14))
    
    # 功率对比图
    ax1.plot(t_plot, p_m_db, label='Excited State (f_m)', alpha=0.7)
    ax1.plot(t_plot, p_g_db, label='Ground State (f_g)', alpha=0.7)
    ax1.set_title("Individual Ridge Powers (dB)")
    ax1.set_ylabel("Power (dB)")
    ax1.legend()
    ax1.grid(True)

    # 差分信号图
    ax2.plot(t_plot, diff_signal, color='purple', label='Joint Diff Indicator')
    ax2.axvline(event_time_ms, color='r', linestyle='--', 
                label=f'Joint Event @ {event_time_ms:.2f} ± {uncertainty:.2f}ms')
    ax2.set_title("Combined Differential Signal (Normalized)")
    ax2.set_ylabel("Amplitude")
    ax2.legend()
    ax2.grid(True)

    # 联合 CUSUM 图
    ax3.plot(t_plot, cusum_joint, color='brown')
    ax3.set_title("Joint CUSUM (Valley detection)")
    ax3.set_xlabel("Time (ms)")
    ax3.set_ylabel("CUSUM Score")
    ax3.grid(True)

    plt.tight_layout()
    plt.show()

    print(f"--- Joint Analysis Result ---")
    print(f"Estimated Decay Time: {event_time_ms:.3f} ms")
    print(f"Uncertainty (1-sigma): ± {uncertainty:.3f} ms")
    print(f"Observed M-G Offset: {time_diff:.3f} ms")
    
    return event_time_ms, uncertainty


# 示例调用 (假设 iq_data 已加载)
# t_event = detect_ground_state_emergence(iq_data)

# 使用示例:
iq_data = np.load('C:/Users/van4w/Desktop/127La/8243_TestModePY82_26-04-07_22-12-25/data_cutInjection/IQ_8243_PY82ch1_0264_trigger_12_2026-04-08T02-01-17.npy') # 你的原始 complex array
f_g=309.8639e6
f_m=309.8622e6
#event_t = analyze_ion(iq_data, fs=5e6, f_center=310e6, f_target=f_m, t_start_ms=500, t_end_ms=800, decay=True)
#print(f"Detected Change Time: {event_t:.2f} ms")
event_t, err = analyze_ion_joint(iq_data, fs=5e6, f_center=310e6, 
        f_m_target=f_m, f_g_target=f_g, t_start_ms=500, t_end_ms=800, f_sweep_range=800)
print(f"Detected Change Time: {event_t:.2f} ms +- {err:.2f} ms")
