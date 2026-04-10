#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np
import pyfftw, multiprocessing, os, sys
from scipy.signal import windows

from preprocessing import Preprocessing
from psd_array import *

def psd_cutInjection(file_folder, file_strs, output_folder, window_length, n_average, overlap_ratio, n_hop=None, window=None, beta=None):
    '''
    Cutting puyuans' raw data based on injection, for each injection saved as a .npz file
    Requirement for the data files:
        only recorded from the puyuan 4-channel new devices
        >= 1 triggers in each file
        if no trigger in one file, the pointer of signal will skip the files until find another file has several triggers.

    file_folder:        .data files' file folder
    file_strs:          .data file list
    output_folder:      output .npz files' folder
    window_length:      length of the tapering window, a.k.a. L
    n_average:          length of average, len k
    overlap_ratio:      the overlap ratio for the L-D points and if K sequences cover the entire N data points
                        tot_N = ( L + D * ( K - 1 ) ) * n_frame
                        overlap_ratio = 1 - D / L
    n_hop:              number of points skipped between each frame, default tot_N
    window:             to be chosen from ["bartlett", "blackman", "hamming", "hanning", "kaiser", etc.] (from scipy.signal.windows)
                        if None, a rectangular window is implied
                        if "kaiser" is given, an additional argument of beta is expected
    ***
    https://ccrma.stanford.edu/~jos/sasp/Welch_s_Method_Windows.html#sec:wwelch
    overlap_ratio should always match the window to reduce side-lobe level
    rectangular window, overlap_ratio = 0, D = L
    non-rectangular window, overlap_ratio >= 0.5 , D <= L / 2
                Hamming, Hanning, and any other generalized Hamming window, overlap_ratio = 0.5
                Blackman window, overlap_ratio = 2/3 , D = L / 3
    '''
    if not file_strs:
        print('No file in the assigned file list, please check and try again.')
        return
    first_extension = None
    try:
        for filename in file_strs:
            full_path = os.path.join(file_folder, filename)
            _, current_extension = os.path.splitext(full_path)
            if first_extension is None:
                first_extension = current_extension
            else:
                # check if all the suffixes are identical
                if current_extension != first_extension:
                    raise ValueError("Error: the suffix of the file '{:}' within the file list is not match!".format(filename))
        if first_extension not in ['.data']:
            raise ValueError("Error: All files in the list bear the suffix {:}, must be .data.".format(first_extension))
    except ValueError as e:
        print("The files in the list do not meet the requirements. {:}".format(e))
        sys.exit(1)

    n_thread = multiprocessing.cpu_count()
    window_sequence = handle_windows(window_length, window, beta)
    # round the padded frame length up to the next radix-2 power
    n_point = window_length
    if window_length == int (window_length * overlap_ratio): overlap_ratio = 0.5
    D = int((1 - overlap_ratio) * window_length) 
    N = int(window_length + D * (n_average - 1))
    # modify hop number
    if n_hop == 0 or n_hop == None: n_hop = N
    if n_hop < 0:
        print("Input Error: n_hop must >= 0!")
        sys.exit(1)
    # read for each files
    file_strs = sorted(file_strs, key=lambda x: int(x.split('_')[1].split('.')[0])) # sorted the files by name
    # create an FFT plan
    dummy = pyfftw.empty_aligned((n_average, window_length))
    fft = pyfftw.builders.fft(dummy, n=n_point, overwrite_input=True, threads=n_thread)
    additional_x, lastTriggerData_remain, trigger_frame, offset = np.array([]), 0, 0, 0
    for i, file_str in enumerate(file_strs):
        bud = Preprocessing(file_folder+file_str, puyuan_new=True, abs_trigger=False)
        ThisFileTimestamp = bud.date_time + np.timedelta64(8, 'h') # convert to '+08' timezone
        if len(bud.trigger_timestamp) == 0:
            print('Warning: no trigger in file {:}, skip it. Continue until another file with more than 1 trigger.'.format(file_st))
            additional_x, lastTriggerData_remain, trigger_frame, offset = np.array([]), 0, 0, 0
            continue
        else:
            for trigger_i, trigger_timestamp in enumerate(bud.trigger_timestamp):
                ThisTriggerData_remain = trigger_timestamp * bud.data_len + lastTriggerData_remain - offset
                if trigger_i == 0 and trigger_frame == 0: 
                    offset = trigger_timestamp * bud.data_len
                    ThisDataTimestamp = ThisFileTimestamp + np.timedelta64(int(offset/bud.sampling_rate), 's')
                    continue
                while True:
                    if ThisTriggerData_remain > N:
                        x = np.hstack((additional_x, bud.load(N-lastTriggerData_remain,offset)[1]))
                        signal = np.lib.stride_tricks.as_strided(x, (n_average, window_length), (x.strides[0] * D, x.strides[0])) * window_sequence
                        if trigger_frame == 0:
                            psd_array = np.mean(np.absolute(np.fft.fftshift(fft(signal), axes=-1))**2 / np.sum(window_sequence**2) / bud.sampling_rate, axis=0)
                        else:
                            psd_array = np.vstack((psd_array, np.mean(np.absolute(np.fft.fftshift(fft(signal), axes=-1))**2 / np.sum(window_sequence**2) / bud.sampling_rate, axis=0)))
                        ThisTriggerData_remain -= n_hop
                        trigger_frame += 1
                        if lastTriggerData_remain > 0:
                            if lastTriggerData_remain - n_hop > 0:
                                offset = 0
                                additional_x = additional_x[n_hop:]
                                lastTriggerData_remain = len(additional_x)
                            else:
                                offset = n_hop - lastTriggerData_remain
                                additional_x = np.array([])
                                lastTriggerData_remain = 0
                        else:
                            offset += n_hop
                    else:
                        frequencies = np.linspace(-bud.sampling_rate/2, bud.sampling_rate/2, n_point+1) # Hz
                        if n_point % 2 == 1: frequencies += bud.sampling_rate / (2*n_point)
                        freq_idx_0, freq_idx_1 = np.searchsorted(frequencies, [-bud.span/2, bud.span/2])
                        frequencies = frequencies[freq_idx_0:freq_idx_1+1]
                        times = np.arange(trigger_frame+1) / bud.sampling_rate * n_hop # s
                        if trigger_i == 0:
                            print('Injection between {:} and {:}'.format(file_strs[i-1], file_str))
                            np.savez(output_folder+file_folder.split('/')[-2].split('_')[0]+'_'+file_str.split('_')[0]+'_{:04d}'.format(int(file_strs[i-1].split('_')[1].split('.')[0]))+'-'+'{:04d}_'.format(int(file_str.split('_')[1].split('.')[0]))+ThisDataTimestamp.astype('datetime64[s]').item().strftime('%Y-%m-%dT%H-%M-%S')+'.npz', frequencies=frequencies, times=times, psd_arrays=psd_array[:,freq_idx_0:freq_idx_1])
                        else:
                            print('{:}, trigger: {:}'.format(file_str, trigger_i))
                            np.savez(output_folder+file_folder.split('/')[-2].split('_')[0]+'_'+file_str.split('_')[0]+'_{:04d}'.format(int(file_str.split('_')[1].split('.')[0]))+'_trigger_{:}_'.format(trigger_i)+ThisDataTimestamp.astype('datetime64[s]').item().strftime('%Y-%m-%dT%H-%M-%S')+'.npz', frequencies=frequencies, times=times, psd_arrays=psd_array[:,freq_idx_0:freq_idx_1])
                        additional_x = np.array([]) 
                        lastTriggerData_remain = 0 
                        offset = trigger_timestamp * bud.data_len
                        ThisDataTimestamp = ThisFileTimestamp + np.timedelta64(int(offset/bud.sampling_rate), 's')
                        trigger_frame = 0
                        break

            ThisTriggerData_remain = bud.n_sample + lastTriggerData_remain - offset
            while True:
                if ThisTriggerData_remain > N:
                    x = np.hstack((additional_x, bud.load(N-lastTriggerData_remain,offset)[1]))
                    signal = np.lib.stride_tricks.as_strided(x, (n_average, window_length), (x.strides[0] * D, x.strides[0])) * window_sequence
                    if trigger_frame == 0:
                        psd_array = np.mean(np.absolute(np.fft.fftshift(fft(signal), axes=-1))**2 / np.sum(window_sequence**2) / bud.sampling_rate, axis=0)
                    else:
                        psd_array = np.vstack((psd_array, np.mean(np.absolute(np.fft.fftshift(fft(signal), axes=-1))**2 / np.sum(window_sequence**2) / bud.sampling_rate, axis=0)))
                    ThisTriggerData_remain -= n_hop
                    trigger_frame += 1
                    offset += n_hop
                else:
                    additional_x = np.hstack((additional_x,bud.load(ThisTriggerData_remain,offset)[1]))
                    lastTriggerData_remain = len(additional_x)
                    offset = 0
                    break
