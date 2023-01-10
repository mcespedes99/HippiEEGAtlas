import pyedflib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import traceback
from file_manager import create_EDF, create_bipolars, create_bipolar_tsv, apply_bipolar_criteria
import argparse
import re
import shutil
import os

input_files = snakemake.input.edf
annot_files = snakemake.input.annot
tsv_file = snakemake.input.tsv
processes = int(snakemake.params.processes)
output_dir = snakemake.output.out_dir
parc_path = snakemake.input.parc
noncon_to_con_tf_path = snakemake.input.tf

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
                print('caca2')
                new_edf ='/tmp/'+re.search(pattern_complete, edf).group()
                if not os.path.exists(new_edf): # REMOVE THE IF
                    shutil.copy(edf, '/tmp/')
            # Build output folder/file for edf file and tsv
            output_path_ieeg = re.search(pattern_dir, edf).group()
            out_file_name = re.search(pattern=pattern_ieeg, string=edf).group()
            out_tsv = re.search(r'sub-\d{3}', edf).group()+'_space-native_SEEGA.tsv'
            if output_dir.endswith('/'):
                output_path_ieeg = output_dir + output_path_ieeg
                os.makedirs(output_path_ieeg)
                out_path_name = output_path_ieeg+out_file_name+'_epoch.edf'
                out_tsv = output_dir + out_tsv
            else:
                output_path_ieeg = output_dir+'/'+output_path_ieeg
                os.makedirs(output_path_ieeg)
                out_path_name = output_path_ieeg+out_file_name+'_epoch.edf'
                out_tsv = output_dir + '/' + out_tsv

            # Extract info about electrodes positions
            elec_pos = pd.read_csv(tsv_file, sep='\t')
            # Create bipolar combinations
            bipolar_channels, bipolar_info_df = create_bipolars(elec_pos, processes)
            print(bipolar_channels)
            # Create tsv file with information about bipolar channels
            print('here')
            # Only run if not generated previously
            if not os.path.exists(out_tsv):
                df_bipolar = create_bipolar_tsv(parc_path, noncon_to_con_tf_path, bipolar_info_df, out_tsv)

            # Discard data from white matter
            bipolar_channels = apply_bipolar_criteria(df_bipolar, bipolar_channels, processes)
            print(bipolar_channels)
            # Here call function to create new EDF file
            if (new_edf == None):
                create_EDF(edf, time_stamps, bipolar_channels, out_path_name, processes)
            else:
                print('aqui')
                create_EDF(new_edf, time_stamps, bipolar_channels, out_path_name, processes)
                print('delete')
                os.remove(new_edf)
except FileNotFoundError:
    raise FileNotFoundError('The *.edf file was not found in the indicated \
        path.')
except BaseException:
    traceback.print_exc()
    f.close()
# file = "/home/mcesped/projects/ctb-akhanf/cfmm-bids/Khan/epi_iEEG/ieeg/bids/sub-002/ses-002/ieeg/sub-002_ses-002_task-full_run-01_ieeg.edf"
