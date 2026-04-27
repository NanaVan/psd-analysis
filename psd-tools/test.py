#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import os
from psd_cut import *

folder = 'C:/Users/van4w/Desktop/127La/8243_TestModePY82_26-04-07_22-12-25/'
output_folder = 'C:/Users/van4w/Desktop/127La/8243_TestModePY82_26-04-07_22-12-25/data_cutInjection/'
prefix = 'PY82ch1_'
file_start = 264
file_end = 264
win_len = 262144
n_average = 4
overlap_ratio = 0.60881
n_hop = 250108

os.makedirs(output_folder, exist_ok=True)
file_strs = [prefix+'{:}.data'.format(file_idx) for file_idx in range(file_start, file_end+1)]
psd_cutInjection(file_folder=folder, file_strs=file_strs, output_folder='C:/Users/van4w/Desktop/127La/8243_TestModePY82_26-04-07_22-12-25/cutInjection/', window_length=win_len, n_average=n_average, overlap_ratio=overlap_ratio, n_hop=n_hop, window='kaiser', beta=14)
data_cutInjection(file_folder=folder, file_strs=file_strs, output_folder=output_folder)

