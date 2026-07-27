#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import csv
import os
import re
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.special import erf
from scipy.signal import butter, filtfilt

from preprocessing import Preprocessing
from reconstruct_spectrum import extract_peaks_log_detect

# ================= 1. 路径与配置参数 =================
# 8243 (PY82)
# reconstruct_folder = '/mnt/nas_DAQRoom/analyzed_data/puyuan82_data/data/8243_TestModePY82_26-04-07_22-12-25/reconstructed/'
# base_folder        = '/mnt/nas_DAQRoom/analyzed_data/puyuan82_data/data/8243_TestModePY82_26-04-07_22-12-25/baseline_cutInjection/'
# raw_folder         = '/mnt/nas_DAQRoom/analyzed_data/puyuan82_data/data/8243_TestModePY82_26-04-07_22-12-25/cutInjection/'
# raw_data_folder    = '/mnt/nas82_2/raw_data/puyuan82_data/Data/8243_TestModePY82_26-04-07_22-12-25/'
# channel_prefix     = 'PY82ch1'
# output_csv         = '8243_reconstruct_statics_ionMean_decay.csv'

# 8251 (PY84)
reconstruct_folder = '/mnt/nas_DAQRoom/analyzed_data/puyuan84_data/data/8251_TestModePY84_26-04-07_22-13-23/reconstructed/'
base_folder        = '/mnt/nas_DAQRoom/analyzed_data/puyuan84_data/data/8251_TestModePY84_26-04-07_22-13-23/baseline_cutInjection/'
raw_folder         = '/mnt/nas_DAQRoom/analyzed_data/puyuan84_data/data/8251_TestModePY84_26-04-07_22-13-23/cutInjection/'
raw_data_folder    = '/mnt/nas82_2/raw_data/puyuan84_data/Data/8251_TestModePY84_26-04-07_22-13-23/'
channel_prefix     = 'PY84ch1'
output_csv         = '8251_reconstruct_statics_ionMean_decay.csv'

fileIdx_range = [0, 799]

# ================= 2. 基础信号处理与数据读取 =================
def parse_npz_filename(filename, folder_path, prefix="PY82ch1"):
    """解析 npz 文件名并转换为 0-indexed trigger 索引及原始 .data 路径"""
    match_single = re.search(rf'_{prefix}_(\d{{4}})_trigger_(\d+)_', filename)
    match_cross  = re.search(rf'_{prefix}_(\d{{4}})-(\d{{4}})_', filename)

    if match_single:
        seq_num = int(match_single.group(1))
        trigger_index = int(match_single.group(2)) - 1  # 转换 1-indexed 为 0-indexed
        return os.path.join(folder_path, f"{prefix}_{seq_num}.data"), None, trigger_index, False
    elif match_cross:
        seq_curr, seq_next = int(match_cross.group(1)), int(match_cross.group(2))
        return (
            os.path.join(folder_path, f"{prefix}_{seq_curr}.data"),
            os.path.join(folder_path, f"{prefix}_{seq_next}.data"),
            None,
            True,
        )
    else:
        return None, None, None, False

def extract_envelope(data, center_freq, bw, fs_val, order=4):
    """IQ 解调提取指定频段包络"""
    t = np.arange(len(data)) / fs_val
    shifted = data * np.exp(-1j * 2 * np.pi * center_freq * t)
    b, a = butter(order, (bw / 2.0) / (0.5 * fs_val), btype='low')
    return np.abs(filtfilt(b, a, shifted.real) + 1j * filtfilt(b, a, shifted.imag))

def load_iq_raw_signal(fname, raw_data_dir, prefix):
    """从原始 .data 加载全量 IQ 切片（含跨文件拼接逻辑）"""
    file_path, next_file_path, trigger_index, is_cross_file = parse_npz_filename(fname, raw_data_dir, prefix)
    if not file_path or not os.path.exists(file_path):
        return None, None, None

    bud = Preprocessing(file_path, puyuan_new=True, abs_trigger=False)
    fs = bud.sampling_rate
    total_triggers = len(bud.trigger_timestamp)

    if is_cross_file or trigger_index is None:
        trigger_index = total_triggers - 1

    start_sample = max(0, int(bud.trigger_timestamp[trigger_index] * bud.data_len))
    if not is_cross_file and trigger_index < total_triggers - 1:
        end_sample = int(bud.trigger_timestamp[trigger_index + 1] * bud.data_len)
    else:
        end_sample = bud.n_sample

    slice_len_samples = end_sample - start_sample
    _, raw_data_curr = bud.load(size=slice_len_samples, offset=start_sample, draw=False)
    raw_times_curr = np.arange(slice_len_samples) / fs

    raw_data, raw_times = raw_data_curr, raw_times_curr

    if is_cross_file and next_file_path and os.path.exists(next_file_path):
        bud_next = Preprocessing(next_file_path, puyuan_new=True, abs_trigger=False)
        end_sample_next = (
            int(bud_next.trigger_timestamp[0] * bud_next.data_len)
            if len(bud_next.trigger_timestamp) > 0
            else bud_next.n_sample
        )
        _, raw_data_next = bud_next.load(size=end_sample_next, offset=0, draw=False)
        raw_times_next = (np.arange(end_sample_next) / fs) + (raw_times_curr[-1] + 1.0 / fs)
        raw_data = np.concatenate([raw_data_curr, raw_data_next])
        raw_times = np.concatenate([raw_times_curr, raw_times_next])

    return raw_data, raw_times, fs

# ================= 3. 精确衰变计算算法 =================
def compute_precise_pair_decay(raw_data, raw_times, fs, parent_freq, daughter_freq):
    """【配对峰】IQ 双频道包络差值 zero-crossing 求解"""
    bandwidth = 1500.0
    parent_env = extract_envelope(raw_data, parent_freq, bandwidth, fs)
    daughter_env = extract_envelope(raw_data, daughter_freq, bandwidth, fs)
    diff_env = daughter_env - parent_env

    search_mask = (raw_times >= 0.35) & (raw_times <= min(0.8, raw_times[-1]))
    search_indices = np.where(search_mask)[0]
    if len(search_indices) == 0:
        return None, None

    diff_in_search = diff_env[search_indices]
    decision_window = int(0.006 * fs)

    found_idx = next(
        (i for i in range(len(diff_in_search) - decision_window)
         if diff_in_search[i] > 0 and np.mean(diff_in_search[i:i + decision_window]) > 0),
        np.argmin(np.abs(diff_in_search))
    )
    zero_cross_idx = search_indices[found_idx]
    decay_time_raw = raw_times[zero_cross_idx]

    # 物理误差估计
    pre_mask = (raw_times >= 0.10) & (raw_times < 0.30)
    post_mask = (raw_times > decay_time_raw + 0.05) & (raw_times <= min(decay_time_raw + 0.15, raw_times[-1]))
    sigma_noise = (
        np.sqrt((np.std(diff_env[pre_mask])**2 + np.std(diff_env[post_mask])**2) / 2.0)
        if (np.any(pre_mask) and np.any(post_mask))
        else np.std(diff_in_search[:int(0.01 * fs)])
    )

    fit_hw = int(0.001 * fs)
    fit_range = slice(max(0, zero_cross_idx - fit_hw), min(len(raw_times), zero_cross_idx + fit_hw))
    if fit_range.stop - fit_range.start > 2:
        slope_K, _ = np.polyfit(raw_times[fit_range], diff_env[fit_range], 1)
        slope_K = abs(slope_K) if abs(slope_K) > 1e-6 else 1.0
    else:
        slope_K = 1.0

    sigma_stat = sigma_noise / slope_K
    sigma_filter = 1.0 / (2.0 * bandwidth)
    sigma_total = np.sqrt(sigma_stat**2 + sigma_filter**2)

    return decay_time_raw, sigma_total

def compute_precise_single_decay(raw_data, raw_times, fs, peak_freq, exist_state, approx_time):
    """【孤立峰】IQ 单频道包络台阶响应拟合 (Step-function fit)"""
    bandwidth = 1500.0
    env = extract_envelope(raw_data, peak_freq, bandwidth, fs)

    # 下采样加速拟合
    ds = int(fs / 10000)  # ~10 kHz 采样点
    t_sub = raw_times[::ds]
    env_sub = env[::ds]

    # 定义 Erf 台阶拟合函数
    if exist_state == 1:
        # 下降台阶：从高平台降至低平台
        def step_func(t, A, B, t0, sigma_t):
            return 0.5 * A * (1 - erf((t - t0) / (np.sqrt(2) * sigma_t))) + B
    else:
        # 上升台阶：从低平台升至高平台
        def step_func(t, A, B, t0, sigma_t):
            return 0.5 * A * (1 + erf((t - t0) / (np.sqrt(2) * sigma_t))) + B

    # 初始参数估算
    p0 = [np.ptp(env_sub), np.min(env_sub), approx_time, 0.005]
    bounds = (
        [0, 0, 0.1, 0.0001],
        [np.inf, np.inf, min(1.0, raw_times[-1]), 0.05]
    )

    try:
        popt, pcov = curve_fit(step_func, t_sub, env_sub, p0=p0, bounds=bounds, maxfev=2000)
        t_event = popt[2]
        err_fit = np.sqrt(np.diag(pcov))[2]
        
        # 考虑滤波器延迟带宽不确定度
        sigma_filter = 1.0 / (2.0 * bandwidth)
        sigma_total = np.sqrt(err_fit**2 + sigma_filter**2)

        return t_event, sigma_total
    except Exception:
        # 若拟合收敛失败，退回为初始粗略时间与常规带宽不确定度
        return approx_time, 1.0 / (2.0 * bandwidth)

# ================= 4. 筛选待处理文件与断点续传 =================
file_pattern = re.compile(r'_(?:(\d{4})_trigger|(\d{4})-\d{4}_)')
_files = [f for f in os.listdir(reconstruct_folder) if f.endswith('.npz')]
reconstruct_files = []
start_num, end_num = min(fileIdx_range), max(fileIdx_range)

for f in _files:
    _match = file_pattern.search(f)
    if _match:
        seq_str = _match.group(1) or _match.group(2)
        if start_num <= int(seq_str) <= end_num:
            reconstruct_files.append(f)

reconstruct_files.sort()
print(f"共找到 {len(reconstruct_files)} 个待处理文件。")

mode = 'w'
processed_files = set()
if os.path.exists(output_csv):
    choice = input(f"检测到 {output_csv} 已存在。是否跳过已处理文件? (yes/no): ").lower()
    if choice in ['yes', 'y']:
        existing_df = pd.read_csv(output_csv)
        if 'filename' in existing_df.columns:
            processed_files = set(existing_df['filename'].unique())
        mode = 'a'
        print(f"跳过 {len(processed_files)} 个已处理文件。")

fieldnames = [
    'peak_pos', 'err_pos', 'sigma', 'err_sigma', 'height_ratio', 'height_ion', 
    'exist_state', 'exist_time', 'err_exist_time', 'valid', 'pair_num', 'filename'
]

if mode == 'w':
    with open(output_csv, 'w', newline='') as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

# ================= 5. 主处理循环 =================
for ii, fname in enumerate(reconstruct_files):
    if fname in processed_files:
        continue

    file_path = os.path.join(reconstruct_folder, fname)
    base_path = os.path.join(base_folder, fname.replace("reconstruct_", "baseline_").replace('.npz', '.npy'))
    raw_path  = os.path.join(raw_folder, fname.replace("reconstruct_", ""))

    try:
        data = np.load(file_path)
        f_arr = data['frequencies']
        p_log = data['psd_log']
        raw_npz = np.load(raw_path)
        p_arr_raw = raw_npz['psd_arrays']
        total_time = raw_npz['times'][-1]
        p_time_interval = raw_npz['times'][1] - raw_npz['times'][0]
        b_log = np.log(np.load(base_path))

        peaks = extract_peaks_log_detect(f_arr, p_log, p_arr_raw, p_time_interval, b_log, snr_factor=6.0)

        if peaks:
            for p in peaks:
                p['pair_num'] = 0
                p['valid'] = 0 if p['exist_state'] == 2 else 1

            pair_counter, used_indices = 0, set()

            # --- A. 寻找并配对 (Pair Detection) ---
            for i in range(len(peaks)):
                if peaks[i]['exist_state'] == 1 and i not in used_indices:
                    for j in range(len(peaks)):
                        if peaks[j]['exist_state'] == 2 and j not in used_indices:
                            time_condition = np.abs(peaks[i]['exist_time'] + peaks[j]['exist_time'] - total_time) <= 2 * p_time_interval
                            pos_condition = (peaks[i]['peak_pos'] < peaks[j]['peak_pos']) and (np.abs(peaks[i]['peak_pos'] - peaks[j]['peak_pos']) <= 80e3)

                            if time_condition and pos_condition:
                                pair_counter += 1
                                peaks[i]['pair_num'], peaks[j]['pair_num'] = pair_counter, pair_counter
                                peaks[i]['valid'], peaks[j]['valid'] = 1, 1
                                used_indices.add(i)
                                used_indices.add(j)
                                break

            # --- B. 读取全量 IQ 信号统一求解 exist_time ---
            raw_iq_data, raw_iq_times, fs = load_iq_raw_signal(fname, raw_data_folder, channel_prefix)

            for i, p in enumerate(peaks):
                st = p['exist_state']

                # 1. 未衰变稳定离子
                if st == 0:
                    p['exist_time'] = total_time
                    p['err_exist_time'] = 0.0

                # 2. 成功配对的衰变/生成对（使用 IQ 差分零交叉）
                elif st in [1, 2] and p['pair_num'] > 0:
                    if st == 1:
                        # 找到对应的配对子核
                        pair_idx = next(j for j, pk in enumerate(peaks) if pk['pair_num'] == p['pair_num'] and pk['exist_state'] == 2)
                        
                        if raw_iq_data is not None:
                            t_dec, err_dec = compute_precise_pair_decay(
                                raw_iq_data, raw_iq_times, fs, p['peak_pos'], peaks[pair_idx]['peak_pos']
                            )
                            if t_dec is not None:
                                p['exist_time'] = t_dec
                                p['err_exist_time'] = err_dec

                                peaks[pair_idx]['exist_time'] = total_time - t_dec
                                peaks[pair_idx]['err_exist_time'] = err_dec
                                print(f"[{fname}] Pair {p['pair_num']}: IQ双核差分精算 t_decay = {t_dec:.6f} s ± {err_dec*1000:.3f} ms")

                # 3. 未配对的孤立衰变/生成峰（IQ 单核包络台阶拟合）
                elif st in [1, 2] and p['pair_num'] == 0:
                    if raw_iq_data is not None:
                        approx_t = p['exist_time'] if st == 1 else (total_time - p['exist_time'])
                        t_event, err_event = compute_precise_single_decay(
                            raw_iq_data, raw_iq_times, fs, p['peak_pos'], st, approx_t
                        )
                        
                        if st == 1:
                            p['exist_time'] = t_event
                        else:
                            p['exist_time'] = total_time - t_event
                        p['err_exist_time'] = err_event
                        print(f"[{fname}] Single Peak ({p['peak_pos']/1e3:.1f}kHz, State {st}): IQ单核拟合 t_event = {t_event:.6f} s ± {err_event*1000:.3f} ms")
                    else:
                        p['err_exist_time'] = p_time_interval

            # --- C. 追加写入 CSV ---
            with open(output_csv, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                for p in peaks:
                    p['filename'] = fname
                    writer.writerow(p)

        if (ii + 1) % 10 == 0 or (ii + 1) == len(reconstruct_files):
            print(f"进度: {ii+1}/{len(reconstruct_files)} - 已处理: {fname}")

    except Exception as e:
        print(f"错误: 处理文件 {fname} 时出错 - {e}")