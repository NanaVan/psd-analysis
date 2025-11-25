#!/usr/bin/env python3
# −*− coding:utf-8 −*−

import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from nptdms import TdmsFile
from typing import List, Dict, Tuple
from datetime import datetime, timedelta
import json, sys, os.path, struct, warnings

def read_sua_header128(header_data):
    '''
    read SUA format file with 128-byte header
    '''
    if len(header_data) < 128:
        raise ValueError("Header data must be at least 128 bytes")
    fields = struct.unpack('<IIIBBHHHIBIBH32s16I', header_data[:128])
    header = {
        'magic': fields[0],         # 0x01DCEF18
        'prt_count': fields[1],
        'packet_len': fields[2],
        'pack_info': fields[3],
        'data_width': fields[4],
        'interleave': fields[5],
        'channel_count': fields[6],
        'channel_id': fields[7],
        'prt_width': fields[8],
        'data_type': fields[9],
        'total_len': fields[10],
        'reserved1': fields[11],
        'reserved2': fields[12],
        'comment': fields[13].decode(errors='ignore').strip('\x00'),
    }
    for i in range(16):
        header[f'ext_field_{i}'] = fields[14 + i]
    return header
data_type_map = {3: np.int8, 5: np.int16, 6: np.int16} # 6: QI pairing, each for int16


class Preprocessing(object):
    '''
    A class for either reading metadata from a .wvh file and importing segments of data from the corresponding .wvd file
    or reading metadata from a .tiq file and importing segments of data from the same file
    An auxiliary function is implemented to inspect the read data in the time domain mainly for signal's amplitude overflow-check
    '''

    n_buffer = 10**5 # maximum number of IQ pairs to be loaded at one time

    def __init__(self, file_str, verbose=True):
        '''
        file_str is a string describing the location of the .wvh file or .wvd file
        '''
        file_abs = os.path.abspath(file_str)
        self.fname = os.path.basename(file_abs)
        self.fpath = os.path.dirname(file_abs)
        if self.fname[-3:].lower() in ["wvh", "wvd"]:
            self.file_format = "wv"
            self.extract_wv()
        elif self.fname[-3:].lower() == "tiq":
            self.file_format = "tiq"
            self.extract_tiq()
        elif self.fname[-4:].lower() == "tdms":
            self.file_format = "tdms"
            self.extract_tdms()
        elif self.fname[-4:].lower() == "data":
            self.file_format = "data"
            self.extract_data()
        else:
            print("Error: unrecognized file format!\nOnly 'wv' (from R&S devices) or 'tiq' (from Tek devices) or 'tdms' (from NI devices) or 'data' (from puyuan devices) file can be accepted.")
            sys.exit()
        if verbose:
            self.display()

    def extract_wv(self):
        '''
        extract the metadata from the .wvh file
        '''
        with open('/'.join((self.fpath, self.fname[:-3]+"wvh"))) as wvh:
            metadata = json.load(wvh)
        self.date_time = np.datetime64(metadata["timestamp"])
        self.data_format = np.dtype(metadata["format"]).newbyteorder(metadata["endian"])
        self.ref_level = metadata["reference level"] # dBm
        self.center_frequency = metadata["center frequency"] # Hz
        self.span = metadata["span"] # Hz
        self.sampling_rate = metadata["sampling rate"] # Hz
        self.n_sample = metadata["number of samples"]
        self.digitizing_depth = metadata["resolution"] # bit
        self.duration = metadata["duration"] # s

    def extract_tiq(self):
        '''
        extract the metadata from the .tiq file
        '''
        with open('/'.join((self.fpath, self.fname))) as tiq:
            self.n_offset = int(tiq.readline().split('"')[1])
            prefix = '{' + tiq.readline().split('"')[1] + '}'
        with open('/'.join((self.fpath, self.fname)), "rb") as tiq:
            root = ET.fromstring(tiq.read(self.n_offset))
        def get_value(key):
            return next(root[0][0].iter(prefix + key)).text
        # since the analyzers are in the same timezone, we use the datetime in a simple way.
        warnings.filterwarnings("ignore", message="no explicit representation of timezones available for np.datetime64")
        self.date_time = np.datetime64(get_value("DateTime"))
        self.data_format = np.dtype(get_value("NumberFormat").lower()).newbyteorder(get_value("Endian").lower())
        self.ref_level = float(get_value("ReferenceLevel")) # dBm
        self.center_frequency = float(get_value("Frequency")) # Hz
        self.span = float(get_value("AcquisitionBandwidth")) # Hz
        self.sampling_rate = float(get_value("SamplingFrequency")) # Hz
        self.n_sample = int(get_value("NumberSamples"))
        self.scaling = float(get_value("Scaling"))
        self.trig_pos = float(get_value("TriggerPosition")) # s

    def extract_tdms(self):
        '''
        extract the metadata from the .tdms file
        '''
        with TdmsFile.open('/'.join((self.fpath, self.fname))) as tdms:
            try:
                self.date_time = tdms['RecordHeader']['datetime'][0]
            except:
                self.date_time = np.datetime64(datetime.strptime(self.fpath.split('IQ_')[1], '%d_%m_%Y_%H_%M_%S') + timedelta(seconds=tdms['RecordHeader']['absolute timestamp'][0]))
            self.span = 1/tdms['RecordHeader']['dt'][0]/1.25 # Hz
            self.sampling_rate = 1/tdms['RecordHeader']['dt'][0] # Hz
            self.gain = tdms['RecordHeader']['gain'][0]
            self.n_sample = 0
            for chunk in tdms.data_chunks():
                self.n_sample += len(chunk['RecordData']['I'])

    def extract_data(self):
        '''
        extract the metadata from the .data file collected by puyuan device
        '''
        self.n_offset = 128 
        with open('/'.join((self.fpath, self.fname)), 'rb') as f:
            header_data = read_sua_header128(f.read(self.n_offset))
        self.packet_len = header_data['packet_len']
        self.data_format = data_type_map.get(header_data['data_type'])
        self.date_time = ''
        self.n_sample = os.path.getsize('/'.join((self.fpath, self.fname))) // self.packet_len * ((self.packet_len - self.n_offset) // self.data_format().itemsize // 2) 
        self.sampling_rate = 31.25e6
        self.span = self.sampling_rate * 0.8
        self.gain = 1.0
        self.center_frequency = 243e6

    def display(self):
        '''
        display all the parameters as a list
        '''
        print("list of information:\n--------------------")
        print("file format\t\t\t" + self.file_format)
        print("name of file\t\t\t" + self.fname)
        print("path to file\t\t\t" + self.fpath)
        print("timestamp in UTC \t\t" + str(self.date_time))
        try: # wv, tiq
            print("data format\t\t\t" + repr(self.data_format))
            print("reference level\t\t\t{:g} dBm".format(self.ref_level))
            print("center frequency\t\t{:g} MHz".format(self.center_frequency*1e-6))
        except AttributeError: 
            pass
        print("span\t\t\t\t{:g} kHz".format(self.span*1e-3))
        print("sampling rate\t\t\t{:g} kHz".format(self.sampling_rate*1e-3))
        print("number of samples\t\t{:d} IQ pairs".format(self.n_sample))
        print("recording duration (actual)\t{:g} s".format(self.n_sample/self.sampling_rate))
        print("--------------------")
        try: # for wv
            print("digitizing depth\t\t{:d} bits".format(self.digitizing_depth))
            print("recording duration (set)\t{:g} s".format(self.duration))
        except AttributeError: 
            try: # for tiq
                print("scaling\t\t\t\t{:g}".format(self.scaling))
                print("trigger position\t\t{:g} s".format(self.trig_pos))
            except AttributeError: # for tdms
                print("gain\t\t\t\t{:.5e}".format(self.gain))
        print("--------------------")

    def load(self, size, offset, decimating_factor=1, draw=False):
        '''
        size:               amount of IQ pairs to be imported
        offset:             amount of IQ pairs to be skipped over
        decimating_factor:  an positive integer by which the data are decimated, the default value 1 means no downsampling
        '''
        size = (self.n_sample-offset)//decimating_factor if size*decimating_factor+offset > self.n_sample else size # crop the excessive request
        times = (np.arange(size)*decimating_factor + offset) / self.sampling_rate # s
        if self.file_format == "wv":
            wvd = np.memmap('/'.join((self.fpath, self.fname[:-3]+"wvd")), dtype=self.data_format, offset=offset*2*self.data_format.itemsize, mode='r')
            data = wvd[:2*size*decimating_factor].reshape(size,2*decimating_factor)[:,:2].flatten().astype(float).view(complex) / (2**(self.digitizing_depth-1) - .5) # V
        elif self.file_format == "tiq": # for tiq
            tiq = np.memmap('/'.join((self.fpath, self.fname)), dtype=self.data_format, offset=self.n_offset+offset*2*self.data_format.itemsize, mode='r')
            data = tiq[:2*size*decimating_factor].reshape(size,2*decimating_factor)[:,:2].flatten().astype(float).view(complex) * self.scaling # V
        elif self.file_format == "data":  # for puyuan SUA data
            # amount of IQ pairs for each packet
            packet_size_for_data = (self.packet_len - self.n_offset) // 2 // self.data_format().itemsize
            packet_start = offset // packet_size_for_data
            packet_count = (offset + size) // packet_size_for_data + 1 - packet_start
            with open('/'.join((self.fpath, self.fname)), 'rb') as sua:
                sua.seek(packet_start*self.packet_len)
                data_buffer = np.frombuffer(sua.read(packet_count*self.packet_len), dtype=self.data_format)
            actual_packet_num = data_buffer.size // (self.packet_len // self.data_format().itemsize)
            data = np.hstack(data_buffer.reshape(actual_packet_num, -1)[:, self.n_offset//self.data_format().itemsize:])[2*(offset-packet_start*packet_size_for_data):2*(size+offset-packet_start*packet_size_for_data)].reshape(size,2)[:,:2].flatten().astype(float).view(complex) * self.gain
        else: # for tdms
            def data_return(_chunk, _offset, _total_size):
                if _offset >= len(_chunk):
                    _offset -= len(_chunk)
                    return _offset, _total_size, []
                elif _total_size >= len(_chunk[_offset:]):
                    _total_size -= len(_chunk[_offset:])
                    return 0, _total_size, _chunk[_offset:]
                else:
                    return 0, 0, _chunk[_offset:_offset+_total_size]
            with TdmsFile.open('/'.join((self.fpath, self.fname))) as tdms:
                I_data, Q_data = [], []
                I_offset, Q_offset, I_total_size, Q_total_size = offset, offset, size*decimating_factor, size*decimating_factor
                for chunk in tdms.data_chunks():
                    I_offset, I_total_size, _I_data = data_return(chunk['RecordData']['I'], I_offset, I_total_size)
                    I_data.append(_I_data)
                    Q_offset, Q_total_size, _Q_data = data_return(chunk['RecordData']['Q'], Q_offset, Q_total_size)
                    Q_data.append(_Q_data)
                    if I_total_size == 0 and Q_total_size == 0:
                        data = (np.hstack(I_data)[::decimating_factor] + 1j * np.hstack(Q_data)[::decimating_factor]) * self.gain # V
                        break
        if draw:
            self.draw(times, data)
        else:
            return times, data # s, V

    def draw(self, t, signal):
        plt.close("all")
        fig, (axr, axi) = plt.subplots(2, 1, sharex=True, sharey=True)
        axr.plot(t, np.real(signal))
        axr.set_ylabel("in phase")
        axr.set_title(self.fname)
        axi.plot(t, np.imag(signal))
        axi.set_xlim([t.min(), t.max()])
        axi.set_xlabel("time [s]")
        axi.set_ylabel("quadrature")
        plt.show()

    def diagnosis(self, n_point=None, draw=True):
        '''
        plot all the data, after downsampling if necessary, in the time domain
        n_point:    maximum data points in the plot, if omitted, n_buffer is replaced in
                    negative means all samples without downsampling
        '''
        if n_point is None:
            n_point = self.n_buffer
        if n_point < 0:
            offset = 0
            while self.n_sample > offset+self.n_buffer:
                self.load(self.n_buffer, offset, 1, draw)
                offset += self.n_buffer
            self.load(self.n_buffer, offset, 1, draw)
        else:
            decimating_factor = self.n_sample // n_point + 1
            return self.load(n_point, 0, decimating_factor, draw)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: {} path/to/file".format(__file__))
        sys.exit()
    preprocessing = Preprocessing(sys.argv[-1])
    preprocessing.diagnosis()
