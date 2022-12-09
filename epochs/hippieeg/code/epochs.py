import pyedflib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import traceback
from file_manager import create_EDF, find
import argparse
import re
import shutil

parser = argparse.ArgumentParser(description='Script to extract epochs\
                                             from a *.edf file.')
parser.add_argument("I", metavar='INPUT',
                    help='Path to subject directory')
parser.add_argument("O", metavar='OUTPUT',
                    help='Path to the output file.')

args = parser.parse_args()

input_file = args.I
output_file = args.O

# Import file:
try:
    # Copy tsv file
    shutil.copyfile(input_file, output_file)
    # Create edf
    # Create input path:
    pattern_del = r'sub-\d{3}_ses-\d{3}_task-(.+)_run-\d{2}_channels.tsv'
    del_str = re.search(pattern=pattern_del, string=input_file).group()
    input_path_ieeg = input_file.replace(del_str,'')
    # Create output path:
    del_str = re.search(pattern=pattern_del, string=input_file).group()
    pattern_ieeg = r'sub-\d{3}_ses-\d{3}_task-(.+)_run-\d{2}_ieeg'
    output_path_ieeg = output_file.replace(del_str,'')
    # Look for edf files:
    list_files = find("*.edf", input_path_ieeg)
    for edf in list_files:
        f = pyedflib.EdfReader(edf)
        # Get labels:
        labels = f.getSignalLabels()

        # Create df with annotations
        annot = f.readAnnotations()
        annot = {
            'Onset': annot[0],
            'Duration': annot[1],
            'event': annot[2]
        }
        annot = pd.DataFrame(annot)

        # Find time stamps where the 'awake trigger' event is happening
        id = annot['event'].str.contains('awake trigger', case=False)
        time_stamps = annot.Onset.to_numpy()[id]

        if time_stamps.size > 0:
            out_file_name = re.search(pattern=pattern_ieeg, string=edf).group()
            if output_path_ieeg.endswith('/'):
                out_path_name = output_path_ieeg+out_file_name+'_epoch.edf'
            else:
                out_path_name = output_path_ieeg+'/'+out_file_name+'_epoch.edf'
            # Here call function
            create_EDF(f, time_stamps, out_path_name)
except FileNotFoundError:
    raise FileNotFoundError('The *.edf file was not found in the indicated \
        path.')
except BaseException:
    traceback.print_exc()
    f.close()
# file = "/home/mcesped/projects/ctb-akhanf/cfmm-bids/Khan/epi_iEEG/ieeg/bids/sub-002/ses-002/ieeg/sub-002_ses-002_task-full_run-01_ieeg.edf"