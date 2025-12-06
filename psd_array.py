#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np
import pyfftw, multiprocessing, os, system
from scipy.signal import windows

from preprocessing import Preprocessing
import matplotlib.pyplot as plt
import matplotlib.colors as colors

def handle_windows(window_length, window=None, beta=None):
    '''
    handling various windows

    window_length:      length of the tapering window
    window:             to be chosen from ["bartlett", "blackman", "hamming", "hanning", "kaiser"]
                        if None, a rectangular window is implied
                        if "kaiser" is given, an additional argument of beta is expected
    '''
    if window is None:
        window_sequence = np.ones(window_length)
    elif window == "kaiser":
        if beta is None:
            raise ValueError("additional argument beta is empty!")
        else:
            window_sequence = windows.kaiser(window_length, beta)
    else:
        window_func = getattr(np, window)
        window_sequence = window_func(window_length)
    return window_sequence


def psd_array_btm(bud, offset, window_length, n_frame, n_hop, padding_ratio=0, window=None, beta=None):
    '''
    Correlation (Blackman-Tukey) Method Spectral Estimation

    bud:                preprocessing object to be estimated (including data array (len N), fs)
    offset:             number of IQ pairs to be skipped over from beginning
    window_length:      length of the tapering window
    n_frame:            number of frames spanning along the time axis
    n_hop:              number of points skipped between each frame
    padding_ratio:      >= 1, ratio of the full frame length after zero padding to the window length
                        note that the final frame length will be rounded up to the next power of base 2
    window:             to be chosen from ["bartlett", "blackman", "hamming", "hanning", "kaiser", etc.] (from scipy.signal.windows)
                        if None, a rectangular window is implied
                        if "kaiser" is given, an additional argument of beta is expected
    '''
    n_thread = multiprocessing.cpu_count()
    window_sequence = handle_windows(window_length, window, beta)
    # round the padded frame length up to the next radix-2 power
    n_point = int( np.power(2, np.ceil(np.log2(window_length*padding_ratio))) ) if padding_ratio >= 1 else window_length
    # modify hop number
    if n_hop == 0: n_hop = window_length
    # build the frequency sequence
    frequencies = np.linspace(-bud.sampling_rate/2, bud.sampling_rate/2, n_point+1) # Hz
    if n_point % 2 == 1: frequencies += bud.sampling_rate / (2*n_point)
    # build the time sequence
    #times = (offset + np.arange(n_frame+1) * n_hop) / bud.sampling_rate # s 
    times = offset / bud.sampling_rate + np.arange(n_frame+1) / bud.sampling_rate * n_hop # s 
    n_dof = 2    
    # load the data in the block-wise
    n_block = bud.n_buffer // window_length
    # placeholders for the transformed spectrogram
    psd_array = np.full((n_frame, n_point), np.nan)
    # create an FFT plan
    dummy = pyfftw.empty_aligned((n_block, window_length))
    fft = pyfftw.builders.fft(dummy, n=window_length, overwrite_input=True, threads=n_thread)
    ifft = pyfftw.builders.ifft(dummy, n=window_length, overwrite_input=True, threads=n_thread)
    fft_1 = pyfftw.builders.fft(dummy, n=n_point, overwrite_input=True, threads=n_thread)
    while n_frame >= n_block:
        # set the signal
        x = bud.load(window_length+n_hop*(n_block-1), offset)[1]
        signal = np.lib.stride_tricks.as_strided(x, (n_block, window_length), (x.strides[0]*n_hop, x.strides[0]))
        index = psd_array[~np.isnan(psd_array[:,0]), 0].size
        # processing
        psd_array[index:index+n_block] = np.fft.fftshift(np.absolute(fft_1(ifft(np.absolute(fft(signal))**2) * window_sequence))) / bud.sampling_rate
        n_frame -= n_block
        if n_frame == 0: break
        offset += n_hop * n_block
    else:
        dummy = pyfftw.empty_aligned((n_frame, window_length))
        fft = pyfftw.builders.fft(dummy, n=window_length, overwrite_input=True, threads=n_thread)
        ifft = pyfftw.builders.ifft(dummy, n=window_length, overwrite_input=True, threads=n_thread)
        fft_1 = pyfftw.builders.fft(dummy, n=n_point, overwrite_input=True, threads=n_thread)
        signal = bud.load(window_length*n_frame, offset)[1].reshape(n_frame, window_length)
        index = psd_array[~np.isnan(psd_array[:,0]), 0].size
        # processing
        psd_array[index:] = np.fft.fftshift(np.absolute(fft_1(ifft(np.absolute(fft(signal))**2) * window_sequence))) / bud.sampling_rate
    return frequencies, times, psd_array, n_dof # Hz, s, V^2/Hz, 1

def psd_array_welch_multiple_files(file_folder, file_strs, offset, window_length, n_average, overlap_ratio, padding_ratio=0, window=None, beta=None):
    '''
    Average Periodogram (Welch) Method Spectral Estimation for multiple .tdms or .data files from NI or puyuan devices
    
    file_folder:        .tdms files' file folder
    file_strs:          .tdms files list
    offset:             number of IQ pairs to be skipped over from beginning
    window_length:      length of the tapering window, a.k.a. L
    n_average:          length of the average, len K
    overlap_ratio:      the overlap ratio for the L-D points and if K sequences cover the entire N data points
                        tot_N = ( L + D * ( K - 1 ) ) * n_frame
                        overlap_ratio = 1 - D / L
    padding_ratio:      >=1, ratio of the full frame length after zero padding to the window length
                        note that the final frame length will be rounded up to the next power of base 2
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
        if first_extension not in ['.tdms', '.data']:
            raise ValueError("Error: All files in the list bear the suffix {:}, neither .tdms nor .data.".format(first_extension))
    except ValueError as e:
        print("The files in the list do not meet the requirements. {:}".format(e))
        sys.exit(1)
        
    n_thread = multiprocessing.cpu_count()
    window_sequence = handle_windows(window_length, window, beta)
    # round the padded frame length up to the next radix-2 power
    n_point = int( np.power(2, np.ceil(np.log2(window_length*padding_ratio))) ) if padding_ratio >= 1 else window_length
    if window_length == int (window_length * overlap_ratio): overlap_ratio = 0.5
    D = int((1 - overlap_ratio) * window_length) 
    N = int(window_length + D * (n_average - 1))
    total_frame, additional_x, _offset = 0, np.array([]), 0
    # read for each files
    file_strs = sorted(file_strs) # sort the files by name
    # create an FFT plan
    dummy = pyfftw.empty_aligned((n_average, window_length))
    fft = pyfftw.builders.fft(dummy, n=n_point, overwrite_input=True, threads=n_thread)
    sampling_rate, center_frequency = 0, 0
    for i, file_str in enumerate(file_strs):
        bud = Preprocessing(file_folder+file_str)
        try:
            if sampling_rate == 0:
                sampling_rate = bud.sampling_rate
                center_frequency = bud.center_frequency
            elif bud.sampling_rate == sampling_rate and bud.center_frequency == center_frequency: # need to check if all the files sampling rate and center frequency are both the same
                pass
            else:
                raise ValueError("sampling rate or center frequency in files is not the same!")
        except Exception as e:
            return
        # crop the excessive frames for one file
        n_frame = (bud.n_sample -  offset) // N
        #print("n_frame: {:}, offset: {:}, N: {:}".format(n_frame, offset, N))
        if n_frame>0: total_frame += n_frame
        if i == 0:
            if n_frame > 0:
                for j in range(n_frame):
                    if j == 0:
                        x = bud.load(N, offset+j*N)[1]
                        signal = np.lib.stride_tricks.as_strided(x, (n_average, window_length), (x.strides[0] * D, x.strides[0])) * window_sequence
                        psd_array = np.mean(np.absolute(np.fft.fftshift(fft(signal), axes=-1))**2 / np.sum(window_sequence**2) / bud.sampling_rate, axis=0)
                    else:
                        x = bud.load(N, offset+j*N)[1]
                        signal = np.lib.stride_tricks.as_strided(x, (n_average, window_length), (x.strides[0] * D, x.strides[0])) * window_sequence
                        psd_array = np.vstack((psd_array, np.mean(np.absolute(np.fft.fftshift(fft(signal), axes=-1))**2 / np.sum(window_sequence**2) / bud.sampling_rate, axis=0)))
                if n_frame * N < bud.n_sample - offset:
                    _offset = bud.n_sample -  offset - n_frame * N
                    additional_x = bud.load(_offset, offset + n_frame*N)[1]
                    offset = N - _offset
                else:
                    additional_x = np.array([])
                    offset = 0
            else:
                _offset = bud.n_sample - offset
                additional_x = bud.load(_offset, offset)[1]
                offset = N - _offset
        else:
            if n_frame > 0:
                if offset != 0:
                    total_frame += 1
                    x = np.hstack((additional_x,bud.load(N - _offset,0)[1])) 
                    signal = np.lib.stride_tricks.as_strided(x, (n_average, window_length), (x.strides[0] * D, x.strides[0])) * window_sequence
                    psd_array = np.vstack((psd_array, np.mean(np.absolute(np.fft.fftshift(fft(signal), axes=-1))**2 / np.sum(window_sequence**2) / bud.sampling_rate, axis=0)))
                for j in range(n_frame):
                    x = bud.load(N, offset+j*N)[1]
                    signal = np.lib.stride_tricks.as_strided(x, (n_average, window_length), (x.strides[0] * D, x.strides[0])) * window_sequence
                    psd_array = np.vstack((psd_array, np.mean(np.absolute(np.fft.fftshift(fft(signal), axes=-1))**2 / np.sum(window_sequence**2) / bud.sampling_rate, axis=0)))
                if n_frame * N < bud.n_sample - offset:
                    _offset = bud.n_sample -  offset - n_frame * N
                    additional_x = bud.load(_offset, offset + n_frame*N)[1]
                    offset = N - _offset
                else:
                    additional_x = np.array([])
                    offset = 0
            elif (bud.n_sample - offset + len(additional_x) // N > 0):
                total_frame += 1
                x = np.hstack((additional_x,bud.load(N - _offset,0)[1])) 
                signal = np.lib.stride_tricks.as_strided(x, (n_average, window_length), (x.strides[0] * D, x.strides[0])) * window_sequence
                try:
                    psd_array = np.vstack((psd_array, np.mean(np.absolute(np.fft.fftshift(fft(signal), axes=-1))**2 / np.sum(window_sequence**2) / bud.sampling_rate, axis=0)))
                except:
                    psd_array = np.mean(np.absolute(np.fft.fftshift(fft(signal), axes=-1))**2 / np.sum(window_sequence**2) / bud.sampling_rate, axis=0)
            else:
                _offset = bud.n_sample - offset
                additional_x = np.hstack((additional_x,bud.load(N -  _offset,0)[1]))
                offset = N - _offset
                
        # build the frequency sequence
        frequencies = np.linspace(-bud.sampling_rate/2, bud.sampling_rate/2, n_point+1) # Hz
        if n_point % 2 == 1: frequencies += bud.sampling_rate / (2*n_point)
        # build the time sequence
        times = offset / bud.sampling_rate + np.arange(total_frame+1) / bud.sampling_rate * N # s 
        # number of freedom
        n_dof = 2
    return frequencies, times, psd_array, n_dof # Hz, s, V^2/Hz, 1


def psd_array_welch(bud, offset, window_length, n_average, overlap_ratio, n_frame, n_hop, padding_ratio=0, window=None, beta=None):
    '''
    Average Periodogram (Welch) Method Spectral Estimation

    bud:                preprocessing object to be estimated (including data array (len N), fs)
    offset:             number of IQ pairs to be skipped over from beginning
    window_length:      length of the tapering window, a.k.a. L
    n_average:          length of the average, len K
    overlap_ratio:      the overlap ratio for the L-D points and if K sequences cover the entire N data points
                        tot_N = ( L + D * ( K - 1 ) ) * n_frame
                        overlap_ratio = 1 - D / L
    n_frame:            number of frames spanning along the time axis
    n_hop:              number of points skipped between each frame
    padding_ratio:      >= 1, ratio of the full frame length after zero padding to the window length
                        note that the final frame length will be rounded up to the next power of base 2
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
    n_thread = multiprocessing.cpu_count()
    window_sequence = handle_windows(window_length, window, beta)
    # round the padded frame length up to the next radix-2 power
    n_point = int( np.power(2, np.ceil(np.log2(window_length*padding_ratio))) ) if padding_ratio >= 1 else window_length
    if window_length == int (window_length * overlap_ratio): overlap_ratio = 0.5
    D = int((1 - overlap_ratio) * window_length) 
    N = int(window_length + D * (n_average - 1))
    # modify hop number
    if n_hop == 0: n_hop = N
    # crop the excessive frames
    if N * n_frame > bud.n_sample - offset or n_frame < 0: n_frame = (bud.n_sample - offset) // N
    # build the frequency sequence
    frequencies = np.linspace(-bud.sampling_rate/2, bud.sampling_rate/2, n_point+1) # Hz
    if n_point % 2 == 1: frequencies += bud.sampling_rate / (2*n_point)
    # build the time sequence
    #times = (offset + np.arange(n_frame+1) * n_hop) / bud.sampling_rate # s 
    times = offset / bud.sampling_rate + np.arange(n_frame+1) / bud.sampling_rate * n_hop # s 
    # number of freedom
    n_dof = 2
    # create an FFT plan
    dummy = pyfftw.empty_aligned((n_average, window_length))
    fft = pyfftw.builders.fft(dummy, n=n_point, overwrite_input=True, threads=n_thread)
    for i in range(n_frame):
        # set signal
        x = bud.load(N, offset+i*n_hop)[1]
        signal = np.lib.stride_tricks.as_strided(x, (n_average, window_length), (x.strides[0] * D, x.strides[0])) * window_sequence
        if i == 0:
            psd_array = np.mean(np.absolute(np.fft.fftshift(fft(signal), axes=-1))**2 / np.sum(window_sequence**2) / bud.sampling_rate, axis=0)
        else:
            psd_array = np.vstack((psd_array, np.mean(np.absolute(np.fft.fftshift(fft(signal), axes=-1))**2 / np.sum(window_sequence**2) / bud.sampling_rate, axis=0)))
    return frequencies, times, psd_array, n_dof # Hz, s, V^2/Hz, 1

def psd_array_multitaper(bud, offset, window_length, n_frame, n_hop, padding_ratio=0, NW=3, Kmax=4):
    '''
    Multitaper Method (MTM)

    bud:                preprocessing object to be estimated (including data array (len N), fs)
    offset:             number of IQ pairs to be skipped over from beginning
    window_length:      length of the tapering window, a.k.a. L
    n_frame:            number of frames spanning along the time axis
    n_hop:              number of points skipped between each frame
    padding_ratio:      >= 1, ratio of the full frame length after zero padding to the window length
                        note that the final frame length will be rounded up to the next power of base 2
    NW:                 standardized half bandwidth, a.k.a. NW
                        2 * NW = BW * fs
    Kmax:               number of DPSS windows to return (order 0 through Kmax-1)
    '''
    n_thread = multiprocessing.cpu_count()
    window_sequence = windows.dpss(window_length, NW, Kmax, return_ratios=False)
    window_sequence = window_sequence.reshape(Kmax, 1, window_length)
    # modify hop number
    if n_hop == 0: n_hop = window_length
    # round the padded frame length up to the next radix-2 power
    n_point = int( np.power(2, np.ceil(np.log2(window_length*padding_ratio))) ) if padding_ratio >= 1 else window_length
    # crop the excessive frames
    if window_length * n_frame > bud.n_sample - offset or n_frame < 0: n_frame = (bud.n_sample - offset) // window_length
    # build the frequency sequence
    frequencies = np.linspace(-bud.sampling_rate/2, bud.sampling_rate/2, n_point+1) # Hz
    if n_point % 2 == 1: frequencies += bud.sampling_rate / (2*n_point)
    # build the time sequence
    #times = (offset + np.arange(n_frame+1) * n_hop) / bud.sampling_rate # s 
    times = offset / bud.sampling_rate + np.arange(n_frame+1) / bud.sampling_rate * n_hop # s 
    # number of degrees of freedom
    n_dof = 2 * Kmax
    # load the data in the block-wise
    n_block = bud.n_buffer // window_length
    # placeholders for the transformed spectrogram
    psd_array = np.full((n_frame, n_point), np.nan)
    # create an FFT plan
    dummy = pyfftw.empty_aligned((Kmax, n_block, window_length))
    fft = pyfftw.builders.fft(dummy, n=n_point, overwrite_input=True, threads=n_thread)
    while n_frame >= n_block:
        # set signal
        x = bud.load(window_length+n_hop*(n_block-1), offset)[1]
        signal = np.lib.stride_tricks.as_strided(x, (n_block, window_length), (x.strides[0]*n_hop, x.strides[0])).reshape(1, n_block, window_length) * window_sequence
        index = psd_array[~np.isnan(psd_array[:,0]), 0].size
        psd_array[index:index+n_block] = np.mean(np.absolute(np.fft.fftshift(fft(signal), axes=-1))**2, axis=0) / bud.sampling_rate
        n_frame -= n_block
        if n_frame == 0: break
        offset += n_hop * n_block
    else:
        dummy = pyfftw.empty_aligned((Kmax, n_frame, window_length))
        fft = pyfftw.builders.fft(dummy, n=n_point, overwrite_input=True, threads=n_thread)
        x = bud.load(window_length+n_hop*(n_block-1), offset)[1]
        signal = np.lib.stride_tricks.as_strided(x, (n_frame, window_length), (x.strides[0]*n_hop, x.strides[0])).reshape(1, n_frame, window_length) * window_sequence
        index = psd_array[~np.isnan(psd_array[:,0]), 0].size
        psd_array[index:] = np.mean(np.absolute(np.fft.fftshift(fft(signal), axes=-1))**2, axis=0) / bud.sampling_rate
    return frequencies, times, psd_array, n_dof # Hz, s, V^2/Hz, 1

def psd_array_adaptive_multitaper(bud, offset, window_length, n_frame, n_hop, padding_ratio=0, NW=3, Kmax=4):
    '''
    Adaptive Multitaper Method (AMTM)

    bud:                preprocessing object to be estimated (including data array (len N), fs)
    offset:             number of IQ pairs to be skipped over from beginning
    window_length:      length of the tapering window, a.k.a. L
    n_frame:            number of frames spanning along the time axis
    n_hop:              number of points skipped between each frame
    padding_ratio:      >= 1, ratio of the full frame length after zero padding to the window length
                        note that the final frame length will be rounded up to the next power of base 2
    NW:                 standardized half bandwidth, a.k.a. NW
                        2 * NW = BW * fs
    Kmax:               number of DPSS windows to return (order 0 through Kmax-1)
    '''
    n_thread = multiprocessing.cpu_count()
    window_sequence, ratio = windows.dpss(window_length, NW, Kmax, return_ratios=True)
    window_sequence = window_sequence.reshape(Kmax, 1, window_length)
    ratio = ratio.reshape(Kmax, 1, 1)
    # modify hop number
    if n_hop == 0: n_hop = window_length
    # round the padded frame length up to the next radix-2 power
    n_point = int( np.power(2, np.ceil(np.log2(window_length*padding_ratio))) ) if padding_ratio >= 1 else window_length
    # crop the excessive frames
    if window_length * n_frame > bud.n_sample - offset or n_frame < 0: n_frame = (bud.n_sample - offset) // window_length
    # build the frequency sequence
    frequencies = np.linspace(-bud.sampling_rate/2, bud.sampling_rate/2, n_point+1) # Hz
    if n_point % 2 == 1: frequencies += bud.sampling_rate / (2*n_point)
    # build the time sequence
    #times = (offset + np.arange(n_frame+1) * n_hop) / bud.sampling_rate # s 
    times = offset / bud.sampling_rate + np.arange(n_frame+1) / bud.sampling_rate * n_hop # s 
    # load the data in the block-wise
    n_block = bud.n_buffer // window_length
    # placeholders for the transformed spectrogram and number of degrees of freedom
    psd_array = np.full((n_frame, n_point), np.nan)
    n_dof = np.empty((n_frame, n_point))
    # create an FFT plan
    dummy = pyfftw.empty_aligned((Kmax, n_block, window_length))
    fft = pyfftw.builders.fft(dummy, n=n_point, overwrite_input=True, threads=n_thread)
    while n_frame >= n_block:
        # set signal
        x = bud.load(window_length+n_hop*(n_block-1), offset)[1]
        signal = np.lib.stride_tricks.as_strided(x, (n_block, window_length), (x.strides[0]*n_hop, x.strides[0])).reshape(1, n_block, window_length) * window_sequence
        index = psd_array[~np.isnan(psd_array[:,0]), 0].size
        eigpsd = np.fft.fftshift(fft(signal), axes=-1)
        # iteration
        psd_array[index:index+n_block] = np.mean(np.absolute(eigpsd[:2])**2, axis=0) / bud.sampling_rate
        while True:
            variance = np.sum(psd_array[index:index+n_block], axis=1, keepdims=True) / n_point
            weight = (psd_array[index:index+n_block] / (ratio*psd_array[index:index+n_block] + (1-ratio)*variance))**2 * ratio
            psd_temp = np.average(np.absolute(eigpsd)**2, axis=0, weights=weight) / bud.sampling_rate
            if np.allclose(psd_temp, psd_array[index:index+n_block], rtol=1e-5, atol=1e-5): break
            psd_array[index:index+n_block] = psd_temp
        n_dof[index:index+n_block] = 2 * np.sum(weight, axis=0)**2 / np.sum(weight**2, axis=0)
        n_frame -= n_block
        if n_frame == 0: break
        offset += n_hop * n_block
    else:
        dummy = pyfftw.empty_aligned((Kmax, n_frame, window_length))
        fft = pyfftw.builders.fft(dummy, n=n_point, overwrite_input=True, threads=n_thread)
        x = bud.load(window_length+n_hop*(n_frame-1), offset)[1]
        signal = np.lib.stride_tricks.as_strided(x, (n_frame, window_length), (x.strides[0]*n_hop, x.strides[0])).reshape(1, n_frame, window_length) * window_sequence
        index = psd_array[~np.isnan(psd_array[:,0]), 0].size
        eigpsd = np.fft.fftshift(fft(signal), axes=-1)
        psd_array[index:] = np.mean(np.absolute(eigpsd[:2])**2, axis=0) / bud.sampling_rate
        while True:
            variance = np.sum(psd_array[index:], axis=1, keepdims=True) / n_point
            weight = (psd_array[index:] / (ratio*psd_array[index:] + (1-ratio)*variance))**2 * ratio
            psd_temp = np.average(np.absolute(eigpsd)**2, axis=0, weights=weight) / bud.sampling_rate
            if np.allclose(psd_temp, psd_array[index:], rtol=1e-5, atol=1e-5): break
            psd_array[index:] = psd_temp
        n_dof[index:] = 2 * np.sum(weight, axis=0)**2 / np.sum(weight**2, axis=0)
    return frequencies, times, psd_array, n_dof # Hz, s, V^2/Hz, 1
