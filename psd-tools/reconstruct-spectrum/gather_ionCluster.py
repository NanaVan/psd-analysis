#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift
import os
import warnings

def cluster_and_plot_v31(df_cluster_path, df_ref_path, sigma_factor=1.0):
    """
    V31: 置信度驱动的聚类认核
    1. 窗口内密度聚类 (去警告版本)
    2. 针对重叠区域进行多假设评估
    3. 基于纵向层级(Rank)和局部密度计算 Confidence
    """
    # 抑制 sklearn 的 Binning 警告
    warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

    # 初始化结果列
    df_cluster = pd.read_csv(df_cluster_path)
    df_ref = pd.read_csv(df_ref_path)
    df_cluster = df_cluster.copy()
    df_cluster['ion_label'] = 'Background'
    df_cluster['confidence'] = 0.0
    
    # 预处理：记录每个点可能的归属候选 (点索引 -> list of matches)
    potential_matches = {i: [] for i in df_cluster.index}
    
    for _, ref in df_ref.iterrows():
        ion_name = ref['ion']
        p_loc = ref['peak_loc'] * 1000  # kHz to Hz
        p_sig = ref['peak_sig'] * 1000  # kHz to Hz
        
        # 认核窗口 (1.0 sigma)
        mask = (df_cluster['peak_pos'] >= p_loc - sigma_factor * p_sig) & \
               (df_cluster['peak_pos'] <= p_loc + sigma_factor * p_sig)
        
        points_in_window = df_cluster[mask]
        
        # 纵向密度聚类评估 (Vertical density clustering)
        if len(points_in_window) > 3:
            y_data = points_in_window[['height_ion']].values
            # 动态调整带宽，避免过小导致 binning 失败，同时关闭 bin_seeding
            bw = max(0.15, (y_data.max() - y_data.min()) / 6) if y_data.max() > y_data.min() else 0.15
            ms = MeanShift(bandwidth=bw, bin_seeding=False)
            ms.fit(y_data)
            labels = ms.labels_
            
            cluster_centers = ms.cluster_centers_
            unique_labels = np.unique(labels)
            
            cluster_info = []
            for l in unique_labels:
                c_mask = (labels == l)
                cluster_info.append({
                    'label': l,
                    'center': cluster_centers[l][0],
                    'count': np.sum(c_mask),
                    'indices': points_in_window.index[c_mask]
                })
            
            # 按高度排序 (高位簇代表高Q可能)
            cluster_info.sort(key=lambda x: x['center'], reverse=True)
            
            for i, info in enumerate(cluster_info):
                # 过滤极小规模的零散点
                if info['count'] < 2: continue
                
                # 计算初始置信度：层级权重 * 局部密度权重
                rank_weight = 1.0 / (i + 1)
                density_weight = info['count'] / len(points_in_window)
                conf = rank_weight * density_weight
                
                for idx in info['indices']:
                    potential_matches[idx].append({
                        'ion': ion_name,
                        'conf': conf,
                        'rank': i,
                        'center': info['center']
                    })

    # 二次评估：处理重叠竞争
    for idx, matches in potential_matches.items():
        if not matches: continue
        
        # 如果有多个竞争者，选择置信度最高的
        # 这里的置信度已经包含了纵向位次信息
        best_match = max(matches, key=lambda x: x['conf'])
        
        df_cluster.at[idx, 'ion_label'] = best_match['ion']
        df_cluster.at[idx, 'confidence'] = best_match['conf']

    # 绘图逻辑
    plt.figure(figsize=(12, 8))
    
    # 背景点
    bg = df_cluster[df_cluster['ion_label'] == 'Background']
    plt.scatter(bg['peak_pos'], bg['height_ion'], c='lightgray', alpha=0.3, s=10, label='Background')
    
    # 离子点 (按 label 自动着色)
    ions = df_cluster[df_cluster['ion_label'] != 'Background']
    unique_ions = ions['ion_label'].unique()
    
    # 使用随机化 Spectral 以增加颜色区分度
    cmap = plt.get_cmap('Spectral')
    colors = [cmap(i) for i in np.linspace(0, 1, len(unique_ions))]
    np.random.shuffle(colors)
    
    for i, ion in enumerate(unique_ions):
        subset = ions[ions['ion_label'] == ion]
        plt.scatter(subset['peak_pos'], subset['height_ion'], 
                    color=colors[i], s=15, label=ion, edgecolors='none')
        
        # 标注标签：放在簇的顶部
        if not subset.empty:
            top_point = subset.loc[subset['height_ion'].idxmax()]
            plt.text(top_point['peak_pos'], top_point['height_ion'] + 0.2, 
                     ion, color=colors[i], fontsize=8, ha='center', alpha=0.8)

    plt.xlabel('peak_pos (Hz)')
    plt.ylabel('height_ion')
    plt.title(f'Ion Clustering V31: Confidence-Driven Evaluation (Sigma={sigma_factor})')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    return df_cluster




data_folder = 'C:/Users/van4w/Desktop/127La/8243_TestModePY82_26-04-07_22-12-25/'
data_file = '8243_reconstruct_statics_ionMean.csv'
core_file = '204Pt_8243_simulation.csv'
df_results = cluster_and_plot_v31(data_folder+data_file, data_folder+core_file, sigma_factor=2)
df_results.to_csv(data_file[:-4]+'_v30_results.csv')
#visualize_ion_clusters(df, pos_range=[309860000, 309870000])
