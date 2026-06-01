#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_folder = 'C:/Users/van4w/Desktop/127La/8243_TestModePY82_26-04-07_22-12-25/'
data_file = '8243_reconstruct_statics_ionMean.csv'
ref_file = '204Pt_8243_simulation.csv'

df_data = pd.read_csv(data_folder+data_file)
df_ref = pd.read_csv(data_folder+ref_file)

# 提取 peak_sig 数据并剔除缺失值
peak_sig_data = df_ref['peak_sig'].dropna().values

# 计算分位数
q50 = df_ref['peak_sig'].quantile(0.31)
q80 = df_ref['peak_sig'].quantile(0.8)

df_ref[df_ref['peak_sig']<=q50].to_csv(data_folder+ref_file[:-4]+'_filtered_peak_sig.csv')


# 开始绘图
plt.figure(figsize=(10, 6))

# bins='auto' 会根据数据分布自动计算组距，你也可以指定具体数字如 bins=50
plt.hist(peak_sig_data, bins=40, color='skyblue', edgecolor='black', alpha=0.7)
# 绘制垂直线
plt.axvline(q50, color='red', linestyle='--', linewidth=2, label=f'50th Percentile: {q50:.2f}')
plt.axvline(q80, color='orange', linestyle='--', linewidth=2, label=f'80th Percentile: {q80:.2f}')

# 添加标题和坐标轴标签
plt.title('Histogram of peak_sig (Simulation)', fontsize=14)
plt.xlabel('peak_sig Value', fontsize=12)
plt.ylabel('Frequency', fontsize=12)

# 添加网格线以增强可读性
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 显示图像
plt.show()
