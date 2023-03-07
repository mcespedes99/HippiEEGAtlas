import pyedflib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import traceback
from file_manager import create_EDF
import argparse
import re
import shutil
import os

input_files = snakemake.input.edf
annot_files = snakemake.input.annot
processes = int(snakemake.params.processes)
output_dir = snakemake.output.out_dir

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
        
        #I'm planning to change this to use the tsv files to check the annotations as
        # loading the edf files takes a while (to make it more efficient)

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
            # Size of file in MB
            size_edf = os.path.getsize(edf)/1000000
            # Copy file to local scratch if possible
            new_edf = None
            if size_edf <= 0.9*int(snakemake.params.mem):
                # Copy file to local scratch
                print('here')
                new_edf ='/tmp/'+re.search(pattern_complete, edf).group()
                if not os.path.exists(new_edf): # REMOVE THE IF
                    shutil.copy(edf, '/tmp/')
            # Build output folder/file for edf file and tsv
            output_path_ieeg = re.search(pattern_dir, edf).group()
            out_file_name = re.search(pattern=pattern_ieeg, string=edf).group()
            output_path_ieeg = os.path.join(output_dir, output_path_ieeg)
            os.makedirs(output_path_ieeg)
            out_path_name = output_path_ieeg+out_file_name+'_epoch.edf'
            print('here2')
            
            # Here call function to create new EDF file
            if (new_edf == None):
                create_EDF(edf, time_stamps, out_path_name, processes)
            else:
                print('aqui')
                create_EDF(new_edf, time_stamps, out_path_name, processes)
                print('delete')
                os.remove(new_edf)
except FileNotFoundError:
    raise FileNotFoundError('The *.edf file was not found in the indicated \
        path.')
except BaseException:
    traceback.print_exc()
    f.close()
# file = "/home/mcesped/projects/ctb-akhanf/cfmm-bids/Khan/epi_iEEG/ieeg/bids/sub-002/ses-002/ieeg/sub-002_ses-002_task-full_run-01_ieeg.edf"
