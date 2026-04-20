#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
import json, os, glob, gzip
from tqdm import tqdm

# --- 配置 (请务必检查路径是否存在) ---
SOURCE_FOLDER = './data_folder'        
BASELINE_FOLDER = './baseline_folder'  
STATS_CSV = 'channel_statistics.csv' 
OUTPUT_JSON_GZ = 'ionFrequencyExtract_by_frequency.json.gz' # 改为压缩格式

# --- 1. 验证路径与获取文件 ---
if not os.path.exists(SOURCE_FOLDER):
    print(f"错误: 找不到路径 {SOURCE_FOLDER}")
    exit()

# 尝试匹配 npz 文件
file_paths = sorted(glob.glob(os.path.join(SOURCE_FOLDER, "*.npz")))
if len(file_paths) == 0:
    print(f"警告: 在 {SOURCE_FOLDER} 中未找到 .npz 文件。请检查后缀或路径。")
    # 如果文件名是大写的 .NPZ，请修改上面的 glob
    exit()

# --- 2. 载入统计阈值 ---
print(f"正在载入统计阈值并处理 {len(file_paths)} 个文件...")
stats_df = pd.read_csv(STATS_CSV)
threshold_mean = stats_df['mean'].values
threshold_std1 = threshold_mean + stats_df['std'].values
threshold_std3 = threshold_mean + 3 * stats_df['std'].values
frequencies = stats_df['frequency'].values.astype(str)

# --- 3. 初始化容器 ---
results = {f: [[], [], []] for f in frequencies}

# --- 4. 逐个文件比对 ---
for file_path in tqdm(file_paths):
    file_name = os.path.basename(file_path)
    baseline_path = os.path.join(BASELINE_FOLDER, "baseline_{:}.npy".format(file_name[:-4]))
    
    try:
        with np.load(file_path) as data:
            spectrum = data['psd_arrays'].mean(axis=0)
            
        if os.path.exists(baseline_path):
            with np.load(baseline_path) as b_data:
                baseline = b_data['psd_arrays'].mean(axis=0)
                spectrum = spectrum - baseline
        
        # 向量化比对
        idx_mean = np.where(spectrum > threshold_mean)[0]
        idx_std1 = np.where(spectrum > threshold_std1)[0]
        idx_std3 = np.where(spectrum > threshold_std3)[0]
        
        for i in idx_mean: results[frequencies[i]][0].append(file_name)
        for i in idx_std1: results[frequencies[i]][1].append(file_name)
        for i in idx_std3: results[frequencies[i]][2].append(file_name)

    except Exception as e:
        print(f"处理 {file_name} 出错: {e}")

# --- 5. 写入压缩 JSON ---
print(f"正在写入压缩文件 {OUTPUT_JSON_GZ}...")
try:
    with gzip.open(OUTPUT_JSON_GZ, 'wt', encoding='utf-8') as f:
        f.write("{\n")
        num_f = len(frequencies)
        for i, f_val in enumerate(frequencies):
            entry = {
                "over_mean": results[f_val][0],
                "over_1std": results[f_val][1],
                "over_3std": results[f_val][2]
            }
            line = f'  "{f_val}": {json.dumps(entry)}'
            if i < num_f - 1:
                line += ","
            f.write(line + "\n")
        f.write("}")
    print("写入完成！")
except OSError as e:
    if e.errno == 28:
        print("错误: 磁盘空间依然不足，即使使用了压缩。请清理磁盘或联系管理员。")
    else:
        print(f"发生错误: {e}")

# 释放内存
del results
