#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.cluster import MeanShift, RadiusNeighborsClassifier
from sklearn.neighbors import RadiusNeighborsTransformer
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def analyze_and_visualize_v18(csv_path):
    # 1. 加载数据
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"读取文件失败: {e}")
        return

    # --- 第一步：初步去噪 (Radius Outlier Removal) ---
    # 使用邻域点数过滤孤立噪声
    # 假设 peak_pos 尺度很大，这里需要一个合理的 radius
    from sklearn.neighbors import NearestNeighbors
    coords = df[['peak_pos', 'height_ratio']].values
    # 标准化以便于在不同量纲下计算距离
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    coords_scaled = scaler.fit_transform(coords)
    
    nn = NearestNeighbors(radius=0.1) # 相对距离阈值
    nn.fit(coords_scaled)
    adj = nn.radius_neighbors_graph(coords_scaled)
    num_neighbors = np.array(adj.sum(axis=1)).flatten()
    df['is_noise'] = num_neighbors < 5 # 邻域点少于5个视为孤立点

    # --- 第二步：MeanShift 团簇识别 ---
    signal_mask = ~df['is_noise']
    df['ion_type'] = -1
    
    if signal_mask.any():
        ion_df = df[signal_mask].copy()
        # 定位高密度核心，bandwidth 400 针对 peak_pos
        ms_initial = MeanShift(bandwidth=400, bin_seeding=True, min_bin_freq=5)
        ion_df['ms_core'] = ms_initial.fit_predict(ion_df[['peak_pos']])

        # --- 第三步：棒状特征筛选 (Geometric Filter) ---
        valid_clusters = []
        for cluster_id in ion_df['ms_core'].unique():
            cluster_data = ion_df[ion_df['ms_core'] == cluster_id]
            
            x_range = cluster_data['peak_pos'].max() - cluster_data['peak_pos'].min()
            y_range = cluster_data['height_ratio'].max() - cluster_data['height_ratio'].min()
            
            # 棒状特征：窄带宽 (x_range < 3000), 有一定高度 (y_range > 0.1)
            if x_range < 3000 and y_range > 0.1:
                valid_clusters.append(cluster_id)
            else:
                # 不符合棒状特征的设为噪声
                df.loc[cluster_data.index, 'is_noise'] = True

        # 更新信号掩码
        ion_df = df[~df['is_noise']].copy()
        
        if not ion_df.empty:
            # --- 第四步：Gap 细分 ---
            # 在保留的棒状区域内，根据 300 的物理间距切分
            ion_df = ion_df.sort_values('peak_pos')
            final_ids = []
            curr_id = 0
            
            pos_vals = ion_df['peak_pos'].values
            for i in range(len(ion_df)):
                if i > 0:
                    if (pos_vals[i] - pos_vals[i-1]) > 300:
                        curr_id += 1
                final_ids.append(curr_id)
            ion_df['gap_id'] = final_ids

            # --- 第五步：纵向二次 MeanShift 识别 ---
            # 针对每一个 gap_id 块，检查是否需要纵向切开（区分信号与底部噪声）
            refined_labels = []
            global_id_offset = 0
            
            for g_id in ion_df['gap_id'].unique():
                block = ion_df[ion_df['gap_id'] == g_id].copy()
                if len(block) > 10: # 点数足够多才做纵向聚类
                    # 针对 height_ratio 进行 MeanShift
                    ms_v = MeanShift(bandwidth=0.05, bin_seeding=True)
                    block['v_label'] = ms_v.fit_predict(block[['height_ratio']])
                    
                    # 识别出高度最高的那个簇作为真正的离子簇，其他的标记为噪声
                    top_cluster = block.groupby('v_label')['height_ratio'].mean().idxmax()
                    
                    # 更新全局 ID：只保留最顶部的簇
                    block.loc[block['v_label'] == top_cluster, 'final_id'] = global_id_offset
                    block.loc[block['v_label'] != top_cluster, 'final_id'] = -1
                    global_id_offset += 1
                else:
                    block['final_id'] = global_id_offset
                    global_id_offset += 1
                
                refined_labels.append(block[['final_id']])

            final_res = pd.concat(refined_labels)
            df.loc[final_res.index, 'ion_type'] = final_res['final_id']

    # 4. 可视化 (高对比度着色)
    plt.figure(figsize=(14, 8), dpi=100)
    valid = df[df['ion_type'] != -1]
    
    if not valid.empty:
        # 使用 turbo 并打乱颜色，防止相邻簇撞色
        num_clusters = int(valid['ion_type'].max() + 1)
        base_colors = cm.turbo(np.linspace(0, 1, 256))
        np.random.seed(42)
        np.random.shuffle(base_colors)
        custom_cmap = cm.colors.ListedColormap(base_colors)
        
        scatter = plt.scatter(valid['peak_pos'], valid['height_ratio'], 
                             c=valid['ion_type'] % 256, cmap=custom_cmap, s=15, alpha=0.8)
        
        plt.title(f"V18 - Five-Step Refined Logic\n(Radius->MS_Core->GeoFilter->Gap->Vertical_MS)")
        # 保持你关注的特定区域视图
        plt.xlim((309.59e6, 309.61e6)) 
        plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.show()

    # 5. 保存结果
    output_path = csv_path.replace('.csv', '_v18.csv')
    df.to_csv(output_path, index=False)
    print(f"分析完成！结果已保存至: {output_path}")
    return df



data_path = 'C:/Users/van4w/Desktop/127La/8243_TestModePY82_26-04-07_22-12-25/8243_reconstruct_statics_filtered.csv'
analyze_and_visualize_v18(data_path)
#visualize_ion_clusters(df, pos_range=[309860000, 309870000])
