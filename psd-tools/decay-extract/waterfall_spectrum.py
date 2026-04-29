#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def plot_waterfall(file_path, fs=5e6, fc=310e6, f_start=309.86e6, f_end=309.87e6, nfft=50000, noverlap=49000):
    """
    读取.npy格式的IQ数据并绘制频谱瀑布图（Waterfall Plot）。
    
    参数:
    file_path: .npy文件路径
    fs: 采样频率 (Hz)
    fc: 中心频率 (Hz)
    nfft: FFT点数
    noverlap: 相邻窗口重叠点数
    """
    # 1. 加载数据
    # 假设数据是复数格式 (complex64/128)
    data = np.load(file_path)
    
    # 2. 计算短时傅里叶变换 (STFT)
    # return_onesided=False 因为是IQ复信号，需要显示负频率
    f, t, Zxx = signal.stft(data, fs=fs, window=('kaiser', 14), nperseg=nfft, 
                            noverlap=noverlap, return_onesided=False)
    print("Time resolution: {:.6f} s".format((nfft-noverlap)/fs))
    print("Frequency resolution: {:.6f} Hz".format(fs/nfft))
    
    # 3. 整理频率轴和数据
    # Shift将负频率移到左侧，正频率移到右侧
    f = np.fft.fftshift(f) + fc
    Zxx = np.fft.fftshift(Zxx, axes=0)

    idx_0, idx_1 = np.searchsorted(f, [f_start, f_end])
    f = f[idx_0:idx_1]
    
    # 转换为功率谱密度 (dB)
    Zxx_db = 10 * np.log10(np.abs(Zxx)**2 + 1e-12) # 防止log(0)
    Zxx_db = Zxx_db[idx_0:idx_1,:]
    
    # 4. 绘图
    plt.figure(figsize=(12, 8))
    
    # 使用pcolormesh绘制瀑布图
    # shading='gouraud' 可以让颜色过渡更平滑
    plt.pcolormesh(f / 1e3, t * 1e3, Zxx_db.T, shading='auto', cmap='viridis')
    
    plt.title(f'Waterfall Spectrum (Center: {fc/1e6} MHz)')
    plt.ylabel('Time (ms)')
    plt.xlabel('Frequency (kHz)')
    plt.colorbar(label='Intensity (dB)')
    
    plt.tight_layout()
    plt.show()

iq_datafile = 'C:/Users/van4w/Desktop/127La/8243_TestModePY82_26-04-07_22-12-25/data_cutInjection/IQ_8243_PY82ch1_0264_trigger_12_2026-04-08T02-01-17.npy'
plot_waterfall(iq_datafile, fs=5e6, fc=310e6)
