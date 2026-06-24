#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_folder = 'C:/Users/van4w/Desktop/127La/8243_TestModePY82_26-04-07_22-12-25/'
data_file = '8243_reconstruct_statics_ionMean.csv'
ref_file = '204Pt_8243_simulation.csv'#_filtered_peak_sig.csv'

df_data = pd.read_csv(data_folder+data_file)
df_ref = pd.read_csv(data_folder+ref_file)

## plot for different state
#states = [(0, 'lightgray', 'stay'), (1, 'tab:orange', 'decay'), (2, 'tab:blue', 'produce')]
#fig, ax = plt.subplots()
#for state_val, color, label_name in states:
#    subset = df_data[(df_data['exist_state'] == state_val)]
#    ax.scatter(subset['peak_pos'], subset['height_ion'], c=color, label=label_name, s=15, alpha=0.6, edgecolors='none')
#ax.set_xlabel('Frequency (Hz)')
#ax.set_ylabel('Log Height')
#ax.legend()
#plt.tight_layout()
#plt.show()

import matplotlib.colors as mcolors
df_pair = df_data[df_data['pair_num']!=0]
num_pairs = len(df_pair) // 2
hues = np.linspace(0, 1, num_pairs, endpoint=False)
pair_colors = []
color_i = 0
for h in hues:
    color_a = mcolors.hsv_to_rgb([h, 0.9, 0.95])
    color_b = mcolors.hsv_to_rgb([h, 0.4, 0.95])
    pair_colors.append((color_a, color_b))
num_ions = len(np.unique(df_ref['ion'].values))
hues = np.linspace(0, 1, num_ions, endpoint=False)
ion_colors = [mcolors.hsv_to_rgb([c, 0.4, 0.95]) for c in hues]

fig, ax = plt.subplots()
ax.scatter(df_data[df_data['exist_state']==0]['peak_pos'], df_data[df_data['exist_state']==0]['height_ion'], color='lightgray', s=30, marker='D', alpha=0.6, edgecolor='none')
for i, group in df_pair.groupby('filename'):
    for j in np.unique(group['pair_num'].values):
        c_a, c_b = pair_colors[color_i]
        if np.abs(np.diff(group[(group['pair_num']==j)]['peak_pos'].values)[0]) <= 80e3:
            ax.scatter(group[(group['pair_num']==j)&(group['exist_state']==1)]['peak_pos'], group[(group['pair_num']==j)&(group['exist_state']==1)]['height_ion'], color=c_a, s=30, marker='o', alpha=0.8, edgecolors='none')
            ax.scatter(group[(group['pair_num']==j)&(group['exist_state']==2)]['peak_pos'], group[(group['pair_num']==j)&(group['exist_state']==2)]['height_ion'], color=c_a, s=30, marker='X', alpha=0.8, edgecolors='none')
            ax.plot(group[(group['pair_num']==j)]['peak_pos'], group[(group['pair_num']==j)]['height_ion'], color=c_a, lw=3, alpha=0.8)
        color_i += 1
for i, (_id, group) in enumerate(df_ref.groupby('ion')):
    ax.vlines(x=group['peak_loc'].values*1e3, ymin=-1, ymax=df_data['height_ion'].values.max(), color=ion_colors[i], linestyle='--', alpha=0.8, lw=1)
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Log Height')
plt.tight_layout()
plt.show()

