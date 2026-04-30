#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
import numpy as np
import pandas as pd
import time
from reconstruct_spectrum import reconstruct_ion_spectrum

def batch_process_to_npy():
    # --- 路径配置 ---
    # raw_folder = 'D:/Schottky/test/puyuan82/8243_TestModePY82_26-04-07_22-12-25/cutInjection/'
    # base_folder = 'D:/Schottky/test/puyuan82/8243_TestModePY82_26-04-07_22-12-25/baseline_cutInjection/'
    # output_folder = 'D:/Schottky/test/puyuan82/8243_TestModePY82_26-04-07_22-12-25/reconstructed/'
    raw_folder = '/mnt/nas_DAQRoom/analyzed_data/puyuan82_data/data/8243_TestModePY82_26-04-07_22-12-25/cutInjection/'
    base_folder = '/mnt/nas_DAQRoom/analyzed_data/puyuan82_data/data/8243_TestModePY82_26-04-07_22-12-25/baseline_cutInjection/'
    output_folder = '/mnt/nas_DAQRoom/analyzed_data/puyuan82_data/data/8243_TestModePY82_26-04-07_22-12-25/reconstructed/'
    csv_path = '/home/imsexp/OnlineDataAnalysisSystem/2026_209Bi_Exp/Analysis/CNN/8243_TestModePY82_CNN_results.csv'
    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"已创建输出目录: {output_folder}")

    # 获取CNN判断好的含有数据的文件名
    csv_df = pd.read_csv(csv_path)
    raw_files = csv_df[(csv_df['label']==1) & (csv_df['confidence']>=0.9)]['filename'].tolist()
    print(f"共找到 {len(raw_files)} 个文件待处理。")

    for i, filename in enumerate(raw_files):
        raw_path = os.path.join(raw_folder, filename)
        
        # 匹配基线文件: baseline_xxx.npy
        base_filename = "baseline_" + filename.replace('.npz', '.npy')
        base_path = os.path.join(base_folder, base_filename)
        
        if not os.path.exists(base_path):
            print(f"跳过: 未找到基线 {base_filename}")
            continue

        try:
            start_time = time.time()
            
            # --- 1. 加载数据 ---
            raw_data = np.load(raw_path)
            psd_raw = np.mean(raw_data['psd_arrays'], axis=0)
            baseline = np.load(base_path)

            # --- 2. 预处理 ---
            log_psd = np.log(psd_raw)
            # 根据你之前的逻辑，这里对 baseline 进行 log 转换
            log_baseline = np.log(baseline)

            # --- 3. 重构谱线 ---
            rebuilt_log_data = reconstruct_ion_spectrum(
                log_psd, 
                log_baseline, 
                k_high=8.0, 
                k_low=1.8
            )

            # --- 4. 计算 pure_signal_log ---
            pure_signal_log = rebuilt_log_data - log_baseline

            # --- 5. 仅保存 pure_signal_log 为 .npy ---
            # 文件名为 pure_8243_PY82ch1_... .npy
            if np.count_nonzero(pure_signal_log) == 0:
                continue
            else:
                save_filename = "reconstruct_" + filename
                save_path = os.path.join(output_folder, save_filename)
            
                np.savez(save_path, frequencies=raw_data['frequencies'][:-1], psd_log=pure_signal_log)
            
                elapsed = time.time() - start_time
                print(f"[{i+1}/{len(raw_files)}] 已保存: {save_filename} ({elapsed:.2f}s)")

        except Exception as e:
            print(f"处理 {filename} 失败: {str(e)}")

    print("\n所有任务已完成。")

if __name__ == "__main__":
    batch_process_to_npy()
