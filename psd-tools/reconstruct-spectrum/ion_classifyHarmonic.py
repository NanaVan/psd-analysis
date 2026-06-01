#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def identify_ions_with_consensus(df_points, df_ions, top_n=3, sim_sigma_ratio=0.75, display=True):
    """
    df_points: 包含 peak_pos, sigma, height_ion, filename (单位Hz)
    df_ions: 包含 ion, Q, rev_freq, harmonic, peak_loc, peak_sig (单位kHz)
    """
    
    # 1. 单位换算 (kHz -> Hz)
    df_ions_hz = df_ions.copy()
    df_ions_hz['peak_loc_hz'] = df_ions_hz['peak_loc'] * 1000
    df_ions_hz['peak_sig_hz'] = df_ions_hz['peak_sig'] * 1000 * sim_sigma_ratio
    df_ions_hz['rev_freq_hz'] = df_ions_hz['rev_freq'] * 1000

    all_matched_results = []

    # 2. 按 filename 分组处理
    for filename, group in df_points.groupby('filename'):
        # 存储当前文件中每个数据点的 Top N 候选
        # 格式: { point_index: DataFrame(top_n_candidates) }
        point_top_candidates = {}
        
        for idx, point in group.iterrows():
            # 计算 Overlap
            diff_sq = (point['peak_pos'] - df_ions_hz['peak_loc_hz'])**2
            combined_sig_sq = point['sigma']**2 + df_ions_hz['peak_sig_hz']**2
            
            overlap = (1.0 / np.sqrt(2 * np.pi * combined_sig_sq)) * \
                      np.exp(-0.5 * diff_sq / combined_sig_sq)
            
            # 计算 Zscore: Overlap * Q^2 * f^2 * exp(height_ion)
            z_scores = overlap * (df_ions_hz['Q']**2) * \
                       (df_ions_hz['rev_freq_hz']**2) * \
                       np.exp(point['height_ion'])
            
            temp_df = df_ions_hz.copy()
            temp_df['z_score'] = z_scores
            temp_df['point_idx'] = idx  # 记录原始数据点索引
            
            # 保留该点的 Top N
            point_top_candidates[idx] = temp_df.nlargest(top_n, 'z_score')
        
        # 将当前文件所有点的候选合在一个大表中
        if not point_top_candidates:
            continue
        file_candidates = pd.concat(point_top_candidates.values())
        
        # 3. 在当前文件中，按离子(ion)进行共识判定
        for ion_name, ion_group in file_candidates.groupby('ion'):
            # 获取该离子在该文件中涉及的所有谐波及其对应的数据点
            # 排除掉同一个点对应同一个离子多个谐波的情况（如果有），取最高分那个
            unique_point_matches = ion_group.sort_values('z_score', ascending=False).drop_duplicates(['point_idx', 'harmonic'])
            
            if len(unique_point_matches['point_idx'].unique()) < 2:
                continue # 至少需要两个不同的数据点支持该离子
            
            # 检查是否有相邻谐波
            # 我们需要找到所有满足“存在相邻者”的谐波记录
            sorted_matches = unique_point_matches.sort_values('harmonic')
            harmonics = sorted_matches['harmonic'].values
            valid_indices = set()
            
            for i in range(len(harmonics) - 1):
                # 如果当前谐波与下一个谐波相邻
                if harmonics[i+1] - harmonics[i] == 1:
                    valid_indices.add(i)
                    valid_indices.add(i+1)
            
            if valid_indices:
                # 提取这些有相邻支撑的记录
                confirmed_matches = sorted_matches.iloc[list(valid_indices)]
                
                # 记录结果
                for _, row in confirmed_matches.iterrows():
                    all_matched_results.append({
                        'filename': filename,
                        'point_idx': row['point_idx'],
                        'ion': row['ion'],
                        'harmonic': row['harmonic'],
                        'z_score': row['z_score'],
                        'peak_theo_pos': row['peak_loc_hz'],
                        'peak_exp_pos': group.loc[row['point_idx'], 'peak_pos'],
                        'peak_exp_pos_err': group.loc[row['point_idx'], 'err_pos'],
                        'peak_exp_sig': group.loc[row['point_idx'], 'sigma'],
                        'peak_exp_sig_err': group.loc[row['point_idx'], 'err_sigma'],
                        'peak_exp_height_ratio': group.loc[row['point_idx'], 'height_ratio'],
                        'peak_exp_height_ion': group.loc[row['point_idx'], 'height_ion'],
                        'peak_exp_exist_state': group.loc[row['point_idx'], 'exist_state'],
                        'peak_exp_exist_time': group.loc[row['point_idx'], 'exist_time']
                    })
    all_matched_results = pd.DataFrame(all_matched_results)

    if display:
        fig, ax = plt.subplots()
        unique_ions = [i for i in sorted(all_matched_results['ion'].unique()) if i != 'Unknown/Noise']
        import matplotlib.colors as mcolors
        colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.XKCD_COLORS.values())#plt.cm.get_cmap('tab20b', len(unique_ions))
        import random
        random.seed(42)
        random.shuffle(colors)
        color_map = {ion: colors[i % len(colors)] for i, ion in enumerate(unique_ions)}
    
        for ion_id, group in all_matched_results.groupby('ion'):
            c = color_map[ion_id]
            ax.scatter(group['peak_exp_pos'], group['peak_exp_height_ion'], color=c, s=12, alpha=0.7, label=ion_id, zorder=1)
            # Reference Line
            #f_theory = .loc[ref_df['ref_id'] == ion_id, 'peak_theo_loc'].values[0]
            #ax.axvline(x=f_theory, color=c, linestyle='--', alpha=0.3, lw=1)

        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Log Height')
        plt.tight_layout()
        plt.show()


    return all_matched_results

data_folder = 'C:/Users/van4w/Desktop/127La/8243_TestModePY82_26-04-07_22-12-25/'
data_file = '8243_reconstruct_statics_ionMean.csv'
ref_file = '204Pt_8243_simulation.csv'

df_data = pd.read_csv(data_folder+data_file)
df_ref = pd.read_csv(data_folder+ref_file)

processed_df = identify_ions_with_consensus(df_data, df_ref)
processed_df.to_csv(data_folder+data_file[:-4]+'_harmonicFiltered_results.csv', index=False)
