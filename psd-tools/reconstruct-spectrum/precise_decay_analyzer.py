#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import csv
import os
import re
import time  # 引入时间模块进行 Benchmark 统计
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.special import erf
from scipy.signal import butter, filtfilt

from preprocessing import Preprocessing
from reconstruct_spectrum import extract_peaks_log_detect

# ================= 1. 路径与配置参数 =================
# # 8251 (PY84)
# reconstruct_folder = '/mnt/nas_DAQRoom/analyzed_data/puyuan84_data/data/8251_TestModePY84_26-04-07_22-13-23/reconstructed/'
# base_folder        = '/mnt/nas_DAQRoom/analyzed_data/puyuan84_data/data/8251_TestModePY84_26-04-07_22-13-23/baseline_cutInjection/'
# raw_folder         = '/mnt/nas_DAQRoom/analyzed_data/puyuan84_data/data/8251_TestModePY84_26-04-07_22-13-23/cutInjection/'
# raw_data_folder    = '/mnt/nas82_2/raw_data/puyuan84_data/Data/8251_TestModePY84_26-04-07_22-13-23/'
# channel_prefix     = 'PY84ch1'
# output_csv         = '8251_reconstruct_statics_ionMean_decay_5.csv'
# 8243 (PY82)
reconstruct_folder = '/mnt/nas_DAQRoom/analyzed_data/puyuan82_data/data/8243_TestModePY82_26-04-07_22-12-25/reconstructed/'
base_folder        = '/mnt/nas_DAQRoom/analyzed_data/puyuan82_data/data/8243_TestModePY82_26-04-07_22-12-25/baseline_cutInjection/'
raw_folder         = '/mnt/nas_DAQRoom/analyzed_data/puyuan82_data/data/8243_TestModePY82_26-04-07_22-12-25/cutInjection/'
raw_data_folder    = '/mnt/nas82_2/raw_data/puyuan82_data/Data/8243_TestModePY82_26-04-07_22-12-25/'
channel_prefix     = 'PY82ch1'
output_csv         = '8243_reconstruct_statics_ionMean_decay_5.csv'

fileIdx_range = [0, 700]
MAX_WORKERS   = 8  # 并行线程数


# ================= 2. 基础信号处理与解析函数 =================
def parse_npz_filename(filename, prefix="PY82ch1"):
    match_single = re.search(rf'_{prefix}_(\d{{4}})_trigger_(\d+)_', filename)
    match_cross  = re.search(rf'_{prefix}_(\d{{4}})-(\d{{4}})_', filename)

    if match_single:
        seq_num = int(match_single.group(1))
        trigger_index = int(match_single.group(2)) - 1
        return seq_num, None, trigger_index, False
    elif match_cross:
        seq_curr, seq_next = int(match_cross.group(1)), int(match_cross.group(2))
        return seq_curr, seq_next, None, True
    else:
        return None, None, None, False

def extract_envelope(data, center_freq, bw, fs_val, order=4):
    t = np.arange(len(data)) / fs_val
    shifted = data * np.exp(-1j * 2 * np.pi * center_freq * t)
    b, a = butter(order, (bw / 2.0) / (0.5 * fs_val), btype='low')
    return np.abs(filtfilt(b, a, shifted.real) + 1j * filtfilt(b, a, shifted.imag))

def is_decayed_in_window(env, fs, start_idx, threshold_ratio=0.3):
    """
    判断粒子在 start_idx 生成后，在后续观测窗口内是否发生了衰变消失
    """
    if start_idx >= len(env) - int(0.05 * fs):
        return False
    
    # 获取生成后的稳态信号均值
    end_idx_stable = min(len(env), start_idx + int(0.1 * fs))
    start_idx_stable = start_idx + int(0.01 * fs)
    if start_idx_stable >= end_idx_stable:
        return False

    stable_signal = env[start_idx_stable:end_idx_stable]
    if len(stable_signal) == 0:
        return False
    baseline_high = np.mean(stable_signal)

    # 检查尾部信号是否显著下降
    tail_signal = np.mean(env[-int(0.05 * fs):])
    return tail_signal < (baseline_high * threshold_ratio)


# ================= 3. 精确衰变计算算法 =================
def compute_precise_chain_decay(raw_data, raw_times, fs, parent_freq, daughter_freq, approx_t):
    """
    基于 A 消失与 B 生成的差分包络过零点计算精确节点时刻 t
    """
    bandwidth = 1500.0
    parent_env = extract_envelope(raw_data, parent_freq, bandwidth, fs)
    daughter_env = extract_envelope(raw_data, daughter_freq, bandwidth, fs)
    diff_env = daughter_env - parent_env

    # 围绕先验时刻 approx_t 展开 0.2s 的搜索窗口
    t_start = max(0.01, approx_t - 0.1)
    t_end = min(raw_times[-1] - 0.01, approx_t + 0.1)
    search_mask = (raw_times >= t_start) & (raw_times <= t_end)
    search_indices = np.where(search_mask)[0]
    
    if len(search_indices) == 0:
        return approx_t, 1.0 / (2.0 * bandwidth)

    diff_in_search = diff_env[search_indices]
    decision_window = int(0.006 * fs)

    found_idx = next(
        (i for i in range(len(diff_in_search) - decision_window)
         if diff_in_search[i] > 0 and np.mean(diff_in_search[i:i + decision_window]) > 0),
        np.argmin(np.abs(diff_in_search))
    )
    zero_cross_idx = search_indices[found_idx]
    decay_time_raw = raw_times[zero_cross_idx]

    # 估算噪声与斜率
    pre_mask = (raw_times >= max(0, decay_time_raw - 0.15)) & (raw_times < decay_time_raw - 0.05)
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
    """
    单信号边缘拟合 (erf 阶跃)
    exist_state == 1 或 消失沿: fit_type='falling'
    exist_state == 2/3 或 生成沿: fit_type='rising'
    """
    bandwidth = 1500.0
    env = extract_envelope(raw_data, peak_freq, bandwidth, fs)

    ds = max(1, int(fs / 10000))
    t_sub = raw_times[::ds]
    env_sub = env[::ds]

    if exist_state == 1:
        def step_func(t, A, B, t0, sigma_t):
            return 0.5 * A * (1 - erf((t - t0) / (np.sqrt(2) * sigma_t))) + B
    else:
        def step_func(t, A, B, t0, sigma_t):
            return 0.5 * A * (1 + erf((t - t0) / (np.sqrt(2) * sigma_t))) + B

    p0 = [np.ptp(env_sub), np.min(env_sub), approx_time, 0.005]
    
    # 动态适应 approx_time 上下界，避免 bounds 冲突和越界
    t0_min = max(0.0, approx_time - 0.2)
    t0_max = min(raw_times[-1], approx_time + 0.2)
    bounds = (
        [0, 0, t0_min, 0.0001],
        [np.inf, np.inf, t0_max, 0.05]
    )

    try:
        popt, pcov = curve_fit(step_func, t_sub, env_sub, p0=p0, bounds=bounds, maxfev=2000)
        t_event = popt[2]
        err_fit = np.sqrt(np.diag(pcov))[2]
        
        sigma_filter = 1.0 / (2.0 * bandwidth)
        sigma_total = np.sqrt(err_fit**2 + sigma_filter**2)

        return t_event, sigma_total
    except Exception:
        return approx_time, 1.0 / (2.0 * bandwidth)


# ================= 4. 单个 Trigger 的处理函数 =================
def process_single_trigger(fname, bud_curr, bud_next_cache, total_triggers, fs):
    seq_curr, seq_next, trigger_index, is_cross_file = parse_npz_filename(fname, prefix=channel_prefix)

    if is_cross_file or trigger_index is None:
        trigger_index = total_triggers - 1

    start_sample = max(0, int(bud_curr.trigger_timestamp[trigger_index] * bud_curr.data_len))
    if not is_cross_file and trigger_index < total_triggers - 1:
        end_sample = int(bud_curr.trigger_timestamp[trigger_index + 1] * bud_curr.data_len)
    else:
        end_sample = bud_curr.n_sample

    slice_len_samples = end_sample - start_sample
    _, raw_data_curr = bud_curr.load(size=slice_len_samples, offset=start_sample, draw=False)
    raw_times_curr = np.arange(slice_len_samples) / fs

    raw_iq_data, raw_iq_times = raw_data_curr, raw_times_curr

    if is_cross_file and bud_next_cache is not None:
        end_sample_next = (
            int(bud_next_cache.trigger_timestamp[0] * bud_next_cache.data_len)
            if len(bud_next_cache.trigger_timestamp) > 0
            else bud_next_cache.n_sample
        )
        _, raw_data_next = bud_next_cache.load(size=end_sample_next, offset=0, draw=False)
        raw_times_next = (np.arange(end_sample_next) / fs) + (raw_times_curr[-1] + 1.0 / fs)
        raw_iq_data = np.concatenate([raw_data_curr, raw_data_next])
        raw_iq_times = np.concatenate([raw_times_curr, raw_times_next])

    file_path = os.path.join(reconstruct_folder, fname)
    base_path = os.path.join(base_folder, fname.replace("reconstruct_", "baseline_").replace('.npz', '.npy'))
    raw_path  = os.path.join(raw_folder, fname.replace("reconstruct_", ""))

    data = np.load(file_path)
    f_arr = data['frequencies']
    p_log = data['psd_log']
    raw_npz = np.load(raw_path)
    p_arr_raw = raw_npz['psd_arrays']
    total_time = raw_npz['times'][-1]
    p_time_interval = raw_npz['times'][1] - raw_npz['times'][0]
    b_log = np.log(np.load(base_path))

    peaks = extract_peaks_log_detect(f_arr, p_log, p_arr_raw, p_time_interval, b_log, snr_factor=6.0)

    if not peaks:
        return []

    for p in peaks:
        p['pair_num'] = 0
        p['valid'] = 1
        p['is_stable'] = False  # 替换 is_stable_in_window，记录产生后是否发生衰变

    # ------------ 1. 精确时间极小化配对 (频差敏感的严格匹配机制) ------------
    pair_counter = 0
    node_t1_pairs = []  # 存储 (idx_A, idx_child, approx_t1, pair_num)
    matched_children = set()  # 记录已被配对的子核，防止重复抢占

    # 频差敏感配对门限参数
    FREQ_DELTA_THRESH = 5000.0   # 5 kHz 频差界限
    NORMAL_TIME_WINDOW = 0.30    # 普通频差允许的最大时间差 (s)
    STRICT_TIME_WINDOW = 0.05    # 大频差(>5kHz)下严格允许的最大时间差 (s)

    # ---- 第一阶段：主衰变链配对 (State 1 -> State 2) ----
    peaks_state1 = [i for i, p in enumerate(peaks) if p['exist_state'] == 1]
    
    for i in peaks_state1:
        p_A = peaks[i]
        t_A_end = p_A.get('exist_time', total_time / 2.0)
        freq_A = p_A['peak_pos']

        best_child_idx = None
        min_time_diff = float('inf')

        # 遍历寻找时间最接近的 State 2 子核候选
        for j, p_child in enumerate(peaks):
            if p_child['exist_state'] == 2 and j not in matched_children and i != j:
                t_child_start = p_child.get('start_time', total_time - p_child.get('exist_time', 0))
                freq_child = p_child['peak_pos']
                
                time_diff = abs(t_A_end - t_child_start)
                delta_freq = abs(freq_A - freq_child)

                # 动态时间阈值判定：频差 > 5kHz 时，必须满足严格的极短时间差
                allowed_window = STRICT_TIME_WINDOW if delta_freq > FREQ_DELTA_THRESH else NORMAL_TIME_WINDOW

                if time_diff < allowed_window and time_diff < min_time_diff:
                    min_time_diff = time_diff
                    best_child_idx = j

        if best_child_idx is not None:
            pair_counter += 1
            node_t1_pairs.append((i, best_child_idx, t_A_end, pair_counter))
            matched_children.add(best_child_idx)

    # ---- 第二阶段：二级衰变链配对 (State 2 -> State 3) ----
    peaks_state2 = [i for i, p in enumerate(peaks) if p['exist_state'] == 2]

    for i in peaks_state2:
        p_A = peaks[i]
        t_A_end = p_A.get('exist_time', total_time / 2.0)
        freq_A = p_A['peak_pos']

        best_child_idx = None
        min_time_diff = float('inf')

        # 寻找时间最接近的 State 3 孙核候选
        for j, p_child in enumerate(peaks):
            if p_child['exist_state'] == 3 and j not in matched_children and i != j:
                t_child_start = p_child.get('start_time', total_time - p_child.get('exist_time', 0))
                freq_child = p_child['peak_pos']

                time_diff = abs(t_A_end - t_child_start)
                delta_freq = abs(freq_A - freq_child)

                allowed_window = STRICT_TIME_WINDOW if delta_freq > FREQ_DELTA_THRESH else NORMAL_TIME_WINDOW

                if time_diff < allowed_window and time_diff < min_time_diff:
                    min_time_diff = time_diff
                    best_child_idx = j

        if best_child_idx is not None:
            pair_counter += 1
            node_t1_pairs.append((i, best_child_idx, t_A_end, pair_counter))
            matched_children.add(best_child_idx)

    # ------------ 2. 精确节点时刻求解 ------------
    t1_exact_dict = {}

    for idx_A, idx_child, approx_t1, p_num in node_t1_pairs:
        p_A = peaks[idx_A]
        p_child = peaks[idx_child]
        p_A['pair_num'] = p_num
        p_child['pair_num'] = p_num

        t1, err_t1 = compute_precise_chain_decay(
            raw_iq_data, raw_iq_times, fs, p_A['peak_pos'], p_child['peak_pos'], approx_t1
        )
        t1_exact_dict[idx_A] = (t1, err_t1)
        p_child['_exact_birth'] = (t1, err_t1)  # 暂存子核精确生成时刻

    # ------------ 3. 各种粒子的存在寿命/时刻赋值 ------------
    for idx, p in enumerate(peaks):
        st = p['exist_state']

        if st == 0:
            # 【从注入开始完全没有发生衰变的离子】
            p['exist_time'] = total_time  # 存活时间等于整个观测窗口
            p['err_exist_time'] = 0.0
            p['is_stable'] = True  # 未发生衰变

        elif st == 1:
            # 【注入后在观测窗口内发生衰变的母核】
            if idx in t1_exact_dict:
                # 成功与子核配对：过零点精确求解衰变时刻 t1
                t_decay, err_decay = t1_exact_dict[idx]
            else:
                # 盲衰变（未看到子核）：下降沿 fitting 拟合衰变时刻
                app_t = p.get('exist_time', total_time / 2.0)
                t_decay, err_decay = compute_precise_single_decay(
                    raw_iq_data, raw_iq_times, fs, p['peak_pos'], 1, app_t
                )

            p['exist_time'] = max(0.0, t_decay)  # 注入点为0，存活时间 Delta_t = t_decay - 0
            p['err_exist_time'] = err_decay
            p['is_stable'] = False  # 发生了衰变

        elif st in [2, 3]:
            # 【由某种离子产生 (State 2) 或由 State 2 衰变产生 (State 3)】
            # 1. 获取精确生成时刻 t_birth (t1)
            if '_exact_birth' in p:
                t_birth, err_birth = p['_exact_birth']
            else:
                # 未配对到的孤立子核：拟合自身上升沿得到生成时刻
                app_birth = total_time - p.get('exist_time', total_time / 2.0)
                app_birth = max(0.01, min(total_time - 0.01, app_birth))
                t_birth, err_birth = compute_precise_single_decay(
                    raw_iq_data, raw_iq_times, fs, p['peak_pos'], 3, app_birth
                )

            # 2. 判断产生之后在剩余窗口内是否再次发生衰变
            bandwidth = 1500.0
            env_sub = extract_envelope(raw_iq_data, p['peak_pos'], bandwidth, fs)
            start_idx = int(t_birth * fs)
            has_decayed = is_decayed_in_window(env_sub, fs, start_idx)

            if has_decayed:
                # 产生后又发生了衰变：拟合下降沿时刻 t_death
                app_death = max(t_birth + 0.01, min(total_time - 0.01, t_birth + 0.5))
                t_death, err_death = compute_precise_single_decay(
                    raw_iq_data, raw_iq_times, fs, p['peak_pos'], 1, app_death
                )
                p['exist_time'] = max(0.0, t_death - t_birth)  # 存活时间 = 消失时刻 - 产生时刻
                p['err_exist_time'] = np.sqrt(err_birth**2 + err_death**2)
                p['is_stable'] = False  # 发生衰变
            else:
                # 产生后贯穿至窗口末尾：存活时间 = 窗口结束时刻 - 产生时刻
                p['exist_time'] = max(0.0, total_time - t_birth)
                p['err_exist_time'] = err_birth
                p['is_stable'] = True  # 未发生衰变

    for p in peaks:
        p['filename'] = fname
        if '_exact_birth' in p:
            del p['_exact_birth']  # 清理临时属性

    return peaks

# 辅助函数：格式化秒数为 HH:MM:SS
def format_time(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ================= 5. 主程序与 Benchmark 统计 =================
if __name__ == '__main__':
    print("正在扫描并对 .data 文件建立倒排索引...")

    file_pattern = re.compile(r'_(?:(\d{4})_trigger|(\d{4})-\d{4}_)')
    all_npz_files = [f for f in os.listdir(reconstruct_folder) if f.endswith('.npz')]
    start_num, end_num = min(fileIdx_range), max(fileIdx_range)

    data_to_npz_map = defaultdict(list)
    total_matched_files = 0

    for fname in all_npz_files:
        match = file_pattern.search(fname)
        if match:
            seq_str = match.group(1) or match.group(2)
            seq_num = int(seq_str)
            if start_num <= seq_num <= end_num:
                data_to_npz_map[seq_num].append(fname)
                total_matched_files += 1

    # 断点续传
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
        'exist_state', 'exist_time', 'err_exist_time', 'valid', 'pair_num', 
        'is_stable', 'filename'
    ]

    if mode == 'w':
        with open(output_csv, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    # 需要处理的 .data 文件清单
    todo_seq_nums = [
        seq for seq, files in sorted(data_to_npz_map.items()) 
        if any(f not in processed_files for f in files)
    ]
    
    total_data_count = len(todo_seq_nums)
    print(f"\n==================== 任务初始化完成 ====================")
    print(f"待处理 .data 文件数 : {total_data_count} 个")
    print(f"包含 trigger 注入点    : {total_matched_files} 个")
    print(f"并行计算线程数        : {MAX_WORKERS}")
    print(f"========================================================\n")

    # Benchmark 变量定义
    global_start_time = time.time()
    completed_data_count = 0
    time_history = []

    for idx, seq_num in enumerate(todo_seq_nums, 1):
        data_start_time = time.time()
        
        npz_list = data_to_npz_map[seq_num]
        unprocessed_npz = [f for f in npz_list if f not in processed_files]

        curr_data_path = os.path.join(raw_data_folder, f"{channel_prefix}_{seq_num}.data")
        if not os.path.exists(curr_data_path):
            print(f"警告 [{idx}/{total_data_count}]: 原始数据缺失，跳过 -> {curr_data_path}")
            continue

        # 1. 读取并解析 .data
        bud_curr = Preprocessing(curr_data_path, puyuan_new=True, abs_trigger=False)
        fs = bud_curr.sampling_rate
        total_triggers = len(bud_curr.trigger_timestamp)

        has_cross = any(parse_npz_filename(f, prefix=channel_prefix)[3] for f in unprocessed_npz)
        bud_next_cache = None
        if has_cross:
            next_data_path = os.path.join(raw_data_folder, f"{channel_prefix}_{seq_num + 1}.data")
            if os.path.exists(next_data_path):
                bud_next_cache = Preprocessing(next_data_path, puyuan_new=True, abs_trigger=False)

        # 2. 多线程并行计算
        batch_results = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_fname = {
                executor.submit(process_single_trigger, fname, bud_curr, bud_next_cache, total_triggers, fs): fname 
                for fname in unprocessed_npz
            }

            for future in as_completed(future_to_fname):
                fname = future_to_fname[future]
                try:
                    res_peaks = future.result()
                    if res_peaks:
                        batch_results.extend(res_peaks)
                except Exception as e:
                    print(f"错误: 线程处理 {fname} 异常 -> {e}")

        # 3. 追加写盘
        if batch_results:
            with open(output_csv, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                for peak_row in batch_results:
                    writer.writerow(peak_row)

        # 4. Benchmark 统计与节点打印
        data_elapsed = time.time() - data_start_time
        time_history.append(data_elapsed)
        completed_data_count += 1
        
        remaining_data_count = total_data_count - completed_data_count
        avg_time_per_data = np.mean(time_history[-10:])  # 最近 10 个文件的移动平均耗时
        eta_seconds = remaining_data_count * avg_time_per_data
        
        pct = (completed_data_count / total_data_count) * 100

        print(
            f"[{completed_data_count}/{total_data_count} | {pct:5.1f}%] "
            f"文件: {channel_prefix}_{seq_num}.data (含{len(unprocessed_npz)}个trigger) | "
            f"用时: {data_elapsed:6.2f}s | "
            f"剩余: {remaining_data_count:3d} 个 | "
            f"预计剩余时间 (ETA): {format_time(eta_seconds)}"
        )

    total_cost_time = time.time() - global_start_time
    print(f"\n全部处理完成！总耗时: {format_time(total_cost_time)}")