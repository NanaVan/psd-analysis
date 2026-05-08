#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os, csv, re
import numpy as np
import pandas as pd

from reconstruct_spectrum import extract_peaks_log_detect

reconstruct_folder = 'C:/Users/van4w/Desktop/127La/8243_TestModePY82_26-04-07_22-12-25/reconstructed/'
base_folder = 'C:/Users/van4w/Desktop/127La/8243_TestModePY82_26-04-07_22-12-25/baseline_cutInjection/'
raw_folder = '/Users/van4w/Desktop/127La/8243_TestModePY82_26-04-07_22-12-25/cutInjection/'

fileIdx_range = [0, 799]
output_csv = '8243_reconstruct_statics.csv'

_files = [f for f in os.listdir(reconstruct_folder) if f.endswith('.npz')]
reconstruct_files = []
start_num, end_num = min(fileIdx_range), max(fileIdx_range)
for f in _files:
    _match = re.search(r'_(\d{4})_trigger', f)
    if _match:
        seq_num = int(_match.group(1))
        if start_num <= seq_num <= end_num:
            reconstruct_files.append(f)
print(f"共找到 {len(reconstruct_files)} 个文件待处理。")

#import matplotlib.pyplot as plt
#for i, filename in enumerate(reconstruct_files):
#    file_path = os.path.join(reconstruct_folder, filename)
#    with np.load(file_path) as data:
#        fig, ax = plt.subplots()
#        ax.plot(data['frequencies']/1e3, np.exp(data['psd_log'])-1)
#        ax.set_xlabel('Frequency [kHz]')
#        ax.set_title(f'{filename}')
#plt.show()

mode = 'w'
processed_files = set()
# 断点续传逻辑
if os.path.exists(output_csv):
    choice = input(f"检测到 {output_csv} 已存在。是否跳过已处理文件? (yes/no): ").lower()
    if choice == 'yes' or choice == 'y':
        # 读取已处理的文件名
        existing_df = pd.read_csv(output_csv)
        if 'filename' in existing_df.columns:
            processed_files = set(existing_df['filename'].unique())
        mode = 'a' # 追加模式
        print(f"已恢复进度，将跳过 {len(processed_files)} 个文件。")
    else:
        print("将覆盖原文件重新开始。")

# 准备写入 CSV
fieldnames = ['peak_pos', 'err_pos', 'sigma', 'err_sigma', 'height_ratio', 'height_ion', 'filename']

# 如果是新建文件，先写表头
if mode == 'w':
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

# 开始循环
for i, file_path in enumerate(reconstruct_files):
    file_path = os.path.join(reconstruct_folder, file_path)
    fname = os.path.basename(file_path)

    base_path = os.path.join(base_folder, fname.replace("reconstruct_", "baseline_").replace('.npz', '.npy'))
    raw_path = os.path.join(raw_folder, fname.replace("reconstruct_", ""))
    
    if fname in processed_files:
        continue
        
    try:
        data = np.load(file_path)
        f_arr = data['frequencies']
        p_log = data['psd_log']
        p_arr_raw = np.load(raw_path)['psd_arrays']
        b_log = np.load(base_path)
        
        #print(f"进度：{fname} 处理中")
        peaks = extract_peaks_log_detect(f_arr, p_log, p_arr_raw, b_log, snr_factor=6.0)
        
        if peaks:
            # 实时追加到 CSV
            with open(output_csv, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                for p in peaks:
                    p['filename'] = fname
                    writer.writerow(p)
        
        if i % 10 == 0:
            print(f"进度: {i+1}/{len(reconstruct_files)} - 已处理: {fname}")
            
    except Exception as e:
        print(f"错误: 处理文件 {fname} 时出错 - {e}")
