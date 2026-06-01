#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

def identify_ions_v61(df, ref_df):
    """
    V61: Hierarchical Group-Based Outlier Rejection
    Logic:
    1. Pre-filter by 3-sigma frequency window.
    2. Calculate ALL possible Z-scores for all candidate ions.
    3. Iteratively:
       a. Assign point to current best candidate.
       b. Calculate log(Z) distribution per ion group.
       c. If point is outlier (Z < mean-1.5*std), demote to next candidate.
    4. Finalize confidence and candidates.
    """
    # 1. Preparation
    ref_df['ref_id'] = ref_df['ion'] + "_H" + ref_df['harmonic'].astype(str)
    ref_sig = ref_df['peak_sig'].values * 1e3
    ref_loc = ref_df['peak_loc'].values * 1e3
    ref_sens = (ref_df['Q'].values**2) * ((ref_df['rev_freq'].values * 1e6)**2)
    
    all_candidate_scores = []
    
    for _, row in df.iterrows():
        p_d, s_d, h_d = row['peak_pos'], row['sigma'], np.exp(row['height_ion'])
        
        # Spatial 3-sigma constraint
        dist = np.abs(p_d - ref_loc)
        mask = dist < (3.5 * ref_sig)
        
        scores = {}
        if np.any(mask):
            valid_indices = np.where(mask)[0]
            var_sum = s_d**2 + ref_sig[valid_indices]**2
            overlap = (1.0 / np.sqrt(2 * np.pi * var_sum)) * np.exp(-(dist[valid_indices]**2) / (2 * var_sum))
            z_vals = overlap * ref_sens[valid_indices] * h_d
            
            for idx, z in zip(valid_indices, z_vals):
                scores[ref_df.loc[idx, 'ref_id']] = z
        
        # Sort candidates by score descending
        all_candidate_scores.append(sorted(scores.items(), key=lambda x: x[1], reverse=True))

    # 2. Iterative Group Rejection
    current_choice_idx = np.zeros(len(df), dtype=int)
    
    for iteration in range(10):
        assignments = []
        for i, candidates in enumerate(all_candidate_scores):
            idx = current_choice_idx[i]
            if idx < len(candidates):
                assignments.append(candidates[idx])
            else:
                assignments.append(('Unknown', 0.0))
        
        temp_df = pd.DataFrame(assignments, columns=['ion', 'z_score'])
        temp_df['log_z'] = np.log10(temp_df['z_score'].replace(0, 1e-10))
        
        changed = False
        for ion_id, group in temp_df.groupby('ion'):
            if ion_id == 'Unknown' or len(group) < 5: continue
            
            # Use log-distribution for physical outliers (strength/pos mismatch)
            mean_z = group['log_z'].mean()
            std_z = group['log_z'].std()
            threshold = mean_z - 1.5 * std_z
            
            outliers = group[group['log_z'] < threshold].index
            if not outliers.empty:
                current_choice_idx[outliers] += 1
                changed = True
        
        if not changed:
            break

    # 3. Finalize Results
    final_ions, final_conf, final_dicts, final_z = [], [], [], []
    
    for i, candidates in enumerate(all_candidate_scores):
        c_idx = current_choice_idx[i]
        total_z = sum([c[1] for c in candidates])
        
        if c_idx < len(candidates) and total_z > 0:
            best_ion, best_z = candidates[c_idx]
            final_ions.append(best_ion)
            final_conf.append(best_z / total_z)
            final_z.append(best_z)
            final_dicts.append({c[0]: round(c[1]/total_z, 3) for c in candidates[:3]})
        else:
            final_ions.append('Unknown/Noise')
            final_conf.append(0.0)
            final_z.append(0.0)
            final_dicts.append({})

    df['ion'] = final_ions
    df['confidence'] = final_conf
    df['candidates'] = final_dicts
    df['z_score'] = final_z
    
    # 4. Frequency Shift Analysis
    high_conf = df[df['confidence'] > 0.8]
    shift_results = []
    for ion_id, group in high_conf.groupby('ion'):
        if ion_id == 'Unknown/Noise': continue
        f_exp = group['peak_pos'].mean()
        f_theory = ref_df.loc[ref_df['ref_id'] == ion_id, 'peak_loc'].values[0] * 1e3
        shift_results.append({
            'ion': ion_id, 
            'shift_hz': f_exp - f_theory, 
            'count': len(group)
        })
    print("\n--- Frequency Shift Analysis (Confidence > 0.8) ---")
    print(pd.DataFrame(shift_results))

    # 5. Visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), gridspec_kw={'height_ratios': [1, 3]})
    
    # Top: Z-Score Distribution
    valid_z = df[df['ion'] != 'Unknown/Noise']['z_score'].astype(float).replace(0, 1e-10)
    ax1.hist(np.log10(valid_z), bins=100, color='gray', alpha=0.7)
    ax1.set_title('V61: Final Z-Score Distribution (Post-Rejection)')
    ax1.set_xlabel('Log10(Z-Score)')

    # Bottom: High-Contrast Scatter Plot
    high_conf = df[df['confidence']>0.02]
    unique_ions = [i for i in sorted(high_conf['ion'].unique()) if i != 'Unknown/Noise']
    import matplotlib.colors as mcolors
    colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.XKCD_COLORS.values())#plt.cm.get_cmap('tab20b', len(unique_ions))
    import random
    random.seed(42)
    random.shuffle(colors)
    color_map = {ion: colors[i % len(colors)] for i, ion in enumerate(unique_ions)}
    #color_map = {ion: colors(i) for i, ion in enumerate(unique_ions)}
    
    for ion_id, group in high_conf.groupby('ion'):
        if ion_id == 'Unknown/Noise':
            ax2.scatter(group['peak_pos'], group['height_ion'], color='lightgray', s=30, alpha=0.1, label='Unknown/Noise')
        else:
            c = color_map[ion_id]
            #ax2.scatter(group['peak_pos'], group['height_ion'], color=c, s=30, alpha=0.7, label=ion_id, zorder=1)
            # Reference Line
            f_theory = ref_df.loc[ref_df['ref_id'] == ion_id, 'peak_loc'].values[0] * 1e3
            ax2.axvline(x=f_theory, color=c, linestyle='--', alpha=0.3, lw=1)
            #ax2.fill_betweenx(y=[df['height_ion'].values.min(), df['height_ion'].values.max()], x1=f_theory-ref_df.loc[ref_df['ref_id'] == ion_id, 'peak_sig'].values[0] * 1e3, x2=f_theory+ref_df.loc[ref_df['ref_id'] == ion_id, 'peak_sig'].values[0] * 1e3, color=c, alpha=0.1)
    ax2.scatter(df['peak_pos'], df['height_ion'], color='lightgray', s=30, alpha=0.2, zorder=0)
    #for ion_id, group in df.groupby('ion'):
    #    ax2.scatter(group['peak_pos'], group['height_ion'], color='lightgray', s=12, alpha=0.7)

    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Log Height')
    plt.tight_layout()
    plt.show()
    
    return df





data_folder = 'C:/Users/van4w/Desktop/127La/8243_TestModePY82_26-04-07_22-12-25/'
data_file = '8243_reconstruct_statics_ionMean.csv'
#ref_file = '204Pt_8243_simulation.csv'
ref_file = '204Pt_8243_simulation_filtered_peak_sig.csv'

df_data = pd.read_csv(data_folder+data_file)
df_ref = pd.read_csv(data_folder+ref_file)

processed_df = identify_ions_v61(df_data, df_ref)
#processed_df.to_csv(data_folder+'statistical_results.csv', index=False)
