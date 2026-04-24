#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os, time

from nonparams_est import NONPARAMS_EST
from reconstruct_spectrum import reconstruct_ion_spectrum

output_folder = 'C:/Users/van4w/Desktop/127La/test_CNN/test/'
file_strs = ['8243_PY82ch1_0008-0009_2026-04-07T22-12-29.npz', '8243_PY82ch1_0008_trigger_15_2026-04-07T22-12-20.npz', '8243_PY82ch1_0008_trigger_16_2026-04-07T22-12-23.npz', '8243_PY82ch1_0008_trigger_17_2026-04-07T22-12-26.npz', '8243_PY82ch1_0079_trigger_15_2026-04-07T23-15-53.npz', '8243_PY82ch1_0079_trigger_16_2026-04-07T23-15-56.npz', '8243_PY82ch1_0079_trigger_17_2026-04-07T23-15-59.npz', '8243_PY82ch1_0079_trigger_1_2026-04-07T23-15-11.npz', '8243_PY82ch1_0079_trigger_2_2026-04-07T23-15-14.npz', '8243_PY82ch1_0079_trigger_3_2026-04-07T23-15-17.npz', '8243_PY82ch1_0079_trigger_4_2026-04-07T23-15-20.npz']

file_idx = 7
data = np.load(os.path.join(output_folder,file_strs[file_idx]))
freq = data['frequencies'][:-1]
psd = np.mean(data['psd_arrays'], axis=0)

#psd = 10 * np.log10(psd)
#mean_val, std_val = np.mean(psd), np.std(psd)
#psd = (psd - mean_val) / std_val
psd = np.log(psd)

time0 = time.time()
baseline = NONPARAMS_EST(psd).pls('BrPLS', l=10**9, ratio=1e-6)
print('baseline estimation: {:.3f} sec'.format(time.time()-time0))
time0 = time.time()
rebuilt_data = reconstruct_ion_spectrum(psd, baseline, k_high=8.0, k_low=1.8)
print('reconstruct spectrum: {:.3f} sec'.format(time.time()-time0))

fig, ax = plt.subplots(3,1, sharex=True)
ax[0].plot(freq, psd)
ax[0].plot(freq, baseline)
ax[0].set_title(file_strs[file_idx])
ax[1].plot(freq, rebuilt_data)
ax[2].plot(freq, rebuilt_data-baseline)
ax[2].set_xlabel('Frequency [Hz]')
plt.show()
