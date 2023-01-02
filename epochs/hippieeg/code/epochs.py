import pyedflib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import traceback
from file_manager import create_EDF, find
import argparse
import re
import shutil
import os

parser = argparse.ArgumentParser(description='Script to extract epochs\
                                             from a *.edf file.')
parser.add_argument("-p", "--processes" ,metavar='PROCESESS',
                    nargs = 1,
                    help='Number of processes per patient.')
parser.add_argument("-i", "--input", metavar='INPUT',
                    nargs = '*',
                    help='Path to subject directory')
parser.add_argument("-a", "--annot", metavar='ANNOTATION',
                    nargs = '*',
                    help='Path to annotation file')
parser.add_argument("-t", "--tsv", metavar='INPUT_TSV',
                    nargs = '*',
                    help='Path to TSV file')
parser.add_argument("-o", "--output" ,metavar='OUTPUT',
                    nargs = 1,
                    help='Path to the output file.')

args = parser.parse_args()

input_files = args.input
annot_files = args.annot
output_dir = args.output[0]
processes = int(args.processes[0])
tsv_file = args.tsv[0]
print(tsv_file)
print(output_dir)
# Import file:
try:
    # Create edf
    # Create pattern to find input file name
    pattern_complete = r'sub-\d{3}_ses-\d{3}_task-(.+)_run-\d{2}_ieeg.edf'
    # Create patterns to write output files
    pattern_dir = r'ses-\d{3}/.+/'
    pattern_ieeg = r'sub-\d{3}_ses-\d{3}_task-(.+)_run-\d{2}_ieeg'
    # Loop through different edf files given by snakebids
    for file_id in range(len(input_files)):
        # Open annotation file to check if any event is in the data
        annot = pd.read_csv(annot_files[file_id], sep='\t')

        # Find time stamps indexes where the 'awake trigger' event is happening
        id = annot['event'].str.contains('awake trigger', case=False)
        # Check if any results are obtained
        check_size = ~id
        # If any event is found, then the pipeline is run
        if ~check_size.all():
            # I could grab the time stamps from the tsv but the ones in the edf have more
            # precision.
            edf = input_files[file_id]
            f = pyedflib.EdfReader(edf)
            # Get labels:
            labels = f.getSignalLabels()
            # Create df with annotations
            onset_list = f.readAnnotations()[0]
            f.close()
            # Find time stamps where the 'awake trigger' event is happening
            time_stamps = onset_list[id]
            # Copy file to local scratch
            shutil.copy(edf, '/tmp/')
            new_edf ='/tmp/'+re.search(pattern_complete, edf).group()
            # Build output folder/file
            output_path_ieeg = re.search(pattern_dir, edf).group()
            out_file_name = re.search(pattern=pattern_ieeg, string=edf).group()
            if output_dir.endswith('/'):
                output_path_ieeg = output_dir + output_path_ieeg
                os.makedirs(output_path_ieeg)
                out_path_name = output_path_ieeg+out_file_name+'_epoch.edf'
            else:
                output_path_ieeg = output_dir+'/'+output_path_ieeg
                os.makedirs(output_path_ieeg)
                out_path_name = output_path_ieeg+out_file_name+'_epoch.edf'
            # Extract info about electrodes positions
            elec_pos = pd.read_csv(tsv_file, sep='\t')
            # Here call function to create new EDF file
            create_EDF(new_edf, time_stamps, elec_pos, out_path_name, processes)
            os.remove(new_edf)
except FileNotFoundError:
    raise FileNotFoundError('The *.edf file was not found in the indicated \
        path.')
except BaseException:
    traceback.print_exc()
    f.close()
# file = "/home/mcesped/projects/ctb-akhanf/cfmm-bids/Khan/epi_iEEG/ieeg/bids/sub-002/ses-002/ieeg/sub-002_ses-002_task-full_run-01_ieeg.edf"
