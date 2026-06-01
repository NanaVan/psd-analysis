#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os, time

from nonparams_est import NONPARAMS_EST
from reconstruct_spectrum import reconstruct_ion_spectrum, extract_peaks_moments_robust, extract_peaks_log_detect
from bayesian_reconstruct import reconstruct_ion_psd

output_folder = 'C:/Users/van4w/Desktop/127La/test_CNN/test/'
file_strs = ['8243_PY82ch1_0008-0009_2026-04-07T22-12-29.npz', '8243_PY82ch1_0008_trigger_15_2026-04-07T22-12-20.npz', '8243_PY82ch1_0008_trigger_16_2026-04-07T22-12-23.npz', '8243_PY82ch1_0008_trigger_17_2026-04-07T22-12-26.npz', '8243_PY82ch1_0079_trigger_15_2026-04-07T23-15-53.npz', '8243_PY82ch1_0079_trigger_16_2026-04-07T23-15-56.npz', '8243_PY82ch1_0079_trigger_17_2026-04-07T23-15-59.npz', '8243_PY82ch1_0079_trigger_1_2026-04-07T23-15-11.npz', '8243_PY82ch1_0079_trigger_2_2026-04-07T23-15-14.npz', '8243_PY82ch1_0079_trigger_3_2026-04-07T23-15-17.npz', '8243_PY82ch1_0079_trigger_4_2026-04-07T23-15-20.npz', '8243_PY82ch1_0068_trigger_3_2026-04-07T23-05-26.npz', '8243_PY82ch1_0264_trigger_12_2026-04-08T02-01-17.npz', '8243_PY82ch1_0062_trigger_11_2026-04-07T23-00-29.npz']

file_idx = -1
data = np.load(os.path.join(output_folder,file_strs[file_idx]))
freq = data['frequencies'][:-1]
psd_avg = np.mean(data['psd_arrays'], axis=0)

psd_avg = np.log(psd_avg)

#fig, ax = plt.subplots()
#ax.plot(freq, np.log(data['psd_arrays'][10]))
#ax.plot(freq, psd_avg)
#ax.set_xlabel('Frequency [Hz]')
#plt.show()

time0 = time.time()
baseline = NONPARAMS_EST(psd_avg).pls('BrPLS', l=10**9, ratio=1e-6)
print('baseline estimation: {:.3f} sec'.format(time.time()-time0))
time0 = time.time()
rebuilt_avg = reconstruct_ion_spectrum(psd_avg, baseline, k_high=8.0, k_low=1.8)
print('reconstruct spectrum: {:.3f} sec'.format(time.time()-time0))
peaks = extract_peaks_log_detect(freq, rebuilt_avg-baseline, data['psd_arrays'], data['times'], baseline, snr_factor=6.0)
#peaks = extract_peaks_moments_robust(freq, rebuilt_avg-baseline, min_rel_height=0.01, dilation_size=80)
#peaks = extract_peaks_with_physical_bounds(freq, rebuilt_avg-baseline, min_rel_height=0.01, dilation_size=80)

#fig, ax = plt.subplots()
#ax.plot(freq, np.log(data['psd_arrays'][10]), alpha=0.5)
#ax.plot(freq, psd_avg, alpha=0.8)
#ax.plot(freq, rebuilt_avg)
#for _peak in peaks:
#    print(_peak['peak_pos'])
#    ax.fill_betweenx([np.log(data['psd_arrays'][0]).min()-1,np.log(data['psd_arrays'][0]).max()+1],_peak['peak_pos']-6*_peak['sigma'],_peak['peak_pos']+6*_peak['sigma'], color='tab:red', alpha=0.6)
#    #ax.fill_betweenx([np.log(data['psd_arrays'][0]).min()-1,np.log(data['psd_arrays'][0]).max()+1],_peak['left_edge_f'],_peak['right_edge_f'], color='tab:red', alpha=0.6)
#ax.set_title(file_strs[file_idx])
plt.show()

#rebuilt_spectra = np.zeros_like(data['psd_arrays'])
#for i, _psd in enumerate(data['psd_arrays']):
#    rebuilt_spectra[i,:] = reconstruct_ion_psd(np.log(_psd), psd_avg, baseline, k_prior=3.0, k_local=5.0)
#spectra = rebuilt_spectra - baseline

#fig, ax = plt.subplots(3,1, sharex=True)
#ax[0].plot(freq, psd_avg)
#ax[0].plot(freq, baseline)
#ax[0].set_title(file_strs[file_idx])
#ax[1].plot(freq, rebuilt_avg)
#ax[2].plot(freq, rebuilt_avg-baseline)
#ax[2].set_xlabel('Frequency [Hz]')
#plt.show()

#fig, ax = plt.subplots(3,1, sharex=True)
#ax[0].plot(freq, np.log(data['psd_arrays'][10]))
#ax[0].plot(freq, baseline)
#ax[0].set_title(file_strs[file_idx])
#ax[1].plot(freq, rebuilt_spectra[10])
#ax[2].plot(freq, spectra[10])
#ax[2].set_xlabel('Frequency [Hz]')
#plt.show()

#fig, ax = plt.subplots()
#waterfall = ax.pcolormesh(data['frequencies'], data['times'], spectra, shading='fading', cmap='viridis')
#ax.set_xlabel('Frequency [Hz]')
#ax.set_ylabel('Time [s]')
#fig.colorbar(waterfall, ax=ax, label='Power Spectral Density [arb. unit]')
#plt.show()
