#!/usr/bin/env python3
# −*− coding:utf-8 −*−

import numpy as np
import sys, multiprocessing, pyfftw

from preprocessing import Preprocessing
import matplotlib.pyplot as plt
import matplotlib.colors as colors

class Processing(Preprocessing):
    '''
    The working horse for data analysis
    '''
    n_thread = multiprocessing.cpu_count()
    
    def __init__(self, file_str):
        '''
        the .wvh file should reside in the same directory where the .wvd file is found
        '''
        try:
            super().__init__(file_str)
        except FileNotFoundError:
            print("Error: cannot find the files!\nPlease check the input file name.")
            sys.exit()


    # extend the function from Processing
    def tfr_st(self, window_length=1024, n_frame=100, n_offset=0, a=20):
        '''
        time-frequency represetation based on stockwell transform
    
        window_length:      length of the frequencies, a.k.a N
        n_frame:            number of frames spanning along the time axis
        n_offset:           number of IQ pairs to be skipped over
        a:                  width of the Gaussian window     
        '''
        # build the frequency sequence
        frequencies = np.fft.fftshift(np.fft.fftfreq(window_length+2, 1/self.sampling_rate))
        frequencies = (frequencies[1:] + frequencies[:-1]) / 2e3 # kHz
        # build the time sequence
        times = (np.arange(n_offset-1, n_offset+n_frame+1)/self.sampling_rate)
        times = ((times[1:] + times[-1]) / 2) # s
        # create an FFT plan
        dummy = pyfftw.empty_aligned(window_length)
        fft = pyfftw.builders.fft(dummy, n=window_length, overwrite_input=True, threads=self.n_thread)
        ifft = pyfftw.builders.ifft(dummy, n=window_length, overwrite_input=True, threads=self.n_thread)
        ST = np.zeros((n_frame, window_length), dtype=np.complex128)
        f_range = np.fft.fftfreq(window_length,1/window_length)
        # processing
        if window_length % 2 != 0:
            for i in range(n_frame): # t
                ne_int, ne_res = (n_offset - (window_length-1)//2 + i) // self.n_sample, (n_offset - (window_length-1)//2 + i) % self.n_sample
                po_int, po_res = (n_offset + (window_length-1)//2 + i) // self.n_sample, (n_offset + (window_length-1)//2 + i) % self.n_sample
                x = np.concatenate((self.load(self.n_sample-ne_res, ne_res)[1],np.concatenate([self.load(self.n_sample,0)[1]]*(po_int-ne_int-1)),self.load(po_res+1,0)[1])) if po_int - ne_int - 1 >= 0 else self.load(window_length, ne_res)[1]
                xx = fft(np.fft.fftshift(x))
                for j in range(1,window_length): # f
                    ST[i,j] = ifft(np.roll(xx, -j) * np.exp(-2*(np.pi*a)**2*np.fft.fftshift(np.fft.fftfreq(window_length,1/window_length))**2/f_range[j]**2))[(window_length-1)//2]
                ST[i,0] = np.mean(x)
        else:
            for i in range(n_frame): # t
                ne_int, ne_res = (n_offset - window_length//2 + i) // self.n_sample, (n_offset - window_length//2 + i) % self.n_sample
                po_int, po_res = (n_offset + window_length//2 + i) // self.n_sample, (n_offset + window_length//2 + i) % self.n_sample
                x = np.concatenate((self.load(self.n_sample-ne_res, ne_res)[1],np.concatenate([self.load(self.n_sample,0)[1]]*(po_int-ne_int-1)),self.load(po_res+1,0)[1])) if po_int - ne_int - 1 >= 0 else self.load(window_length, ne_res)[1]
                xx = fft(np.fft.fftshift(x))
                for j in range(1,window_length): # f
                    ST[i,j] = ifft(np.fft.fftshift(np.roll(xx, -j) * np.exp(-2*(np.pi*a)**2*np.fft.fftshift(np.fft.fftfreq(window_length,1/window_length))**2/f_range[j]**2), axes=-1))[window_length//2]
                ST[i,0] = np.mean(x)
        ST = np.fft.ifftshift(ST, axes=-1)
        return frequencies, times, ST

    def tfr_winWidOpt_st(self, ):
        return




    
    def tfr_st_psd_phase(self, window_length=1024, n_frame=100, n_offset=0, a=20):
        '''
        time-frequency represetation based on stockwell transform
    
        window_length:      length of the frequencies, a.k.a N
        n_frame:            number of frames spanning along the time axis
        n_offset:           number of IQ pairs to be skipped over
        a:                  width of the Gaussian window     
        '''
        frequencies, times, ST = self.tfr_st(window_length, n_frame, n_offset, a)
        psd = np.absolute(ST)**2 / self.sampling_rate
        phase = np.angle(ST)
        return frequencies, times, psd, phase
    
    def tfr_st_average(self, window_length=1024, n_frame=100, n_average=10, n_offset=0, a=20):
        '''
        time-frequency represetation based on stockwell transform
    
        window_length:      length of the frequencies, a.k.a N
        n_frame:            number of frames spanning along the time axis
        n_average:          number of samples for one average
        n_offset:           number of IQ pairs to be skipped over
        a:                  width of the Gaussian window     
        '''
        frequencies, times, ST = self.tfr_st(window_length, n_frame*n_average, n_offset, a)
        ST = np.mean(ST.reshape(n_frame, n_average, window_length), axis=1)
        return frequencies, times[::n_average], ST
    
    def tfr_st_average_psd_phase(self, window_length=1024, n_frame=100, n_average=10, n_offset=0, a=20):
        '''
        time-frequency represetation based on stockwell transform
    
        window_length:      length of the frequencies, a.k.a N
        n_frame:            number of frames spanning along the time axis
        n_average:          number of samples for one average
        n_offset:           number of IQ pairs to be skipped over
        a:                  width of the Gaussian window     
        '''
        frequencies, times, ST = self.tfr_st_average(window_length, n_frame, n_average, n_offset, a)
        psd = np.absolute(ST)**2 / self.sampling_rate
        phase = np.angle(ST)
        return frequencies, times, psd, phase


#if __name__ == "__main__":
#file_folder = './'
#file_str = 'sinusoid.wvd'
file_folder = "I:/data/2018-12/"
file_str = "20181228_065610.wvd"

bud = Processing(file_folder+file_str)
frequencies, times, psd, phase = bud.tfr_st_psd_phase(window_length=1024, n_frame=1, n_offset=2000, a=20)
freq = (frequencies[1:]+frequencies[:-1])/2

fig, ax = plt.subplots()
ax.semilogy(freq, psd[0])
#pcm = ax.pcolormesh(frequencies, times, psd, norm=colors.LogNorm(vmin=psd.min(), vmax=psd.max()))
#ax.set_xlim([frequencies[0],frequencies[-1]]) # Hz
#ax.set_ylim([times[0], times[-1]])
#cax = fig.colorbar(pcm, ax=ax)
plt.show()  

