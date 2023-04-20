# Temporal file to resize edf files manually

import pyedflib
import numpy as np
import shutil
from data_manager import create_EDF, extract_channel_data
from functools import partial
from multiprocessing.pool import Pool
import sys

def data_crop(signal, new_len):
    signal_cropped = signal[0:new_len]
    return signal_cropped

edf_file = str(sys.argv[1])
new_len = 54000
out_edf = str(sys.argv[2])
processes = 16

# Check if len of edf is equal to max_length
edf_in = pyedflib.EdfReader(edf_file)

# Extract labels 
chn_labels = edf_in.getSignalLabels()
chn_lists = range(len(chn_labels)) 
edf_in.close()
# Extract data
with Pool(processes=processes) as pool2:
    signal = pool2.map(partial(extract_channel_data, edf_file=edf_file,
                                    chn_list=chn_labels), chn_lists)
# Cropping
with Pool(processes=processes) as pool2:
    signal_cropped = pool2.map(partial(data_crop, new_len=new_len), signal)
# Create EDF file
create_EDF(edf_file, out_edf, processes, signal=signal_cropped)