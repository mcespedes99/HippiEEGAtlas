import pyedflib
import numpy as np
import shutil
from data_manager import create_EDF, extract_channel_data
from functools import partial
from multiprocessing.pool import Pool

# Function to zero pad
def data_padding(signal, N_pad):
    zeros = np.zeros(N_pad)
    signal_padded = np.concatenate([signal, zeros])
    return signal_padded

# Function to crop the signals
def data_crop(signal, new_len):
    signal_cropped = signal[0:new_len]
    return signal_cropped

# Function to get data from the edf file
def get_data(edf_file, processes):
    edf_in = pyedflib.EdfReader(edf_file)
    # Extract labels 
    chn_labels = edf_in.getSignalLabels()
    chn_lists = range(len(chn_labels)) 
    # Close edf
    edf_in.close()
    # Extract data
    with Pool(processes=processes) as pool2:
        signal = pool2.map(partial(extract_channel_data, edf_file=edf_file,
                                        chn_list=chn_labels), chn_lists)
    return signal

def main():
    edf_file = snakemake.input.edf
    files_length_csv = snakemake.input.max_len_csv
    out_edf = snakemake.output.out_edf
    processes = int(snakemake.threads)
    new_length = snakemake.params.new_length

    # Read input edf
    edf_in = pyedflib.EdfReader(edf_file)
    # Get number of samples
    N = edf_in.getNSamples()[0]
    edf_in.close()
    # Read csv as array
    files_length = np.genfromtxt(files_length_csv, delimiter='\t')

    # Option A: Zero pad to max length
    if new_length == 'max':
        # Extract max length
        max_length = int(files_length.max())
        if int(N) == max_length:
            # Just copy the file
            shutil.copy(edf_file, out_edf)
        else:
            # Get the data from the original edf
            signal = get_data(edf_file, processes)
            # Zero pad the signal
            N_pad = max_length-N
            # print(N_pad)
            with Pool(processes=processes) as pool2:
                signal_padded = pool2.map(partial(data_padding, N_pad=N_pad), signal)
            # Create EDF file
            create_EDF(edf_file, out_edf, processes, signal=signal_padded)
    
    # Option B: crop signals to min length
    elif new_length == 'min':
        # Extract min length
        min_length = int(files_length.min())
        if int(N) == min_length:
            # Just copy the file
            shutil.copy(edf_file, out_edf)
        else:
            # Get the data from the original edf
            signal = get_data(edf_file, processes)
            # Crop the signal
            with Pool(processes=processes) as pool2:
                signal_cropped = pool2.map(partial(data_crop, new_len=min_length), signal)
            # Create EDF file
            create_EDF(edf_file, out_edf, processes, signal=signal_cropped)
    else:
        raise ValueError('snakemake.params.new_length has a non-valid value.')
if __name__=="__main__":
    main()