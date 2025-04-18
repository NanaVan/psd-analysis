# psd-analysis
a toolkit for spectrum analysis

## Prerequisites
- `Python 3`
- `numpy`, `pyfftw`, `multiprocessing`
- optional: `matplotlib`(used for plotting), `nptdms`(used for parsing .tdms file)

## Usage
Class `Preprocessing` in `preprocessing.py` provides function for displaying the header of data file and extracting data.
- `extract_[?]`: extract the information from the header of data file. `wv` for `.wvd` data file from R&S data recorder, `.tiq` data file from Tektronix spectrum analyzer, `.tdms` data file from NTCAP.
- `display`: display the header information.
- `load`: quick access the metadata.
- `draw`: plot the IQ data from the metadata.
- `diagnosis`: plot all data file for diagonsis.

Functions in `psd.py` are used for the spectrum analysis (2-D). Return frequency, psd, degree of freedom.
- `psd_btm`: Correlation (Blackman-Tukey) Method Spectral Estimation
- `psd_welch`: Average Periodogram (Welch) Method Spectral Estimation
- `psd_multitaper`: Multitaper Method (MTM)
- `psd_adaptive_multitaper`: Adaptive Multitaper Method (AMTM)

Functions in `psd_array.py` are used for the spectrum analysis (3-D). Return frequency, time, psd_array, degree of freedom.
- `psd_array_btm`: Correlation (Blackman-Tukey) Method Spectral Estimation
- `psd_array_welch`: Average Periodogram (Welch) Method Spectral Estimation
- `psd_array_multitaper`: Multitaper Method (MTM)
- `psd_array_adaptive_multitaper`: Adaptive Multitaper Method (AMTM)

## Example
```Python
from preprocessing import Preprocessing
from psd import *
from psd_array import *

bud = Preprocessing('./path/to/file/test_datafile.tiq') # loading the `.tiq` data
frequencies, psd_welch_result, _ = psd_welch(bud, offset=5000, window_length=4096, average=100, overlap_ratio=0.61, padding_ratio=0, window='kaiser', beta=4) # return frequencies, psd using welch method
frequencies, times, psd_array_welch_result, _ = psd_array_welch(bud, offset=5000, window_length=4096, n_average=100, n_frame=100, overlap_ratio=0.5, padding_ratio=0, window='kaiser', beta=4) # return frequencies, time, psd_array using welch method
``` 
