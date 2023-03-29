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

edf = snakemake.input.edf
annot_file = snakemake.input.annot
processes = int(snakemake.params.processes)
out_edf = snakemake.output.out_edf

# Import file:
try:
    # Open annotation file to extract the important events
    annot = pd.read_csv(annot_file, sep='\t')
    
    # Find time stamps indexes where the event is happening
    id = annot['event'].str.contains('awake trigger', case=False)

    # I could grab the time stamps from the tsv but the ones in the edf have more
    # precision.
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
        file_name = os.path.basename(edf)
        new_edf = os.path.join('/tmp',file_name)
        if not os.path.exists(new_edf): # REMOVE THE IF
            shutil.copy(edf, new_edf)
    
    # Here call function to create new EDF file
    if (new_edf == None):
        create_EDF(edf, time_stamps, out_edf, processes)
    else:
        print('aqui')
        create_EDF(new_edf, time_stamps, out_edf, processes)
        print('delete')
        os.remove(new_edf)
except FileNotFoundError:
    raise FileNotFoundError('The *.edf file was not found in the indicated \
        path.')
except BaseException:
    traceback.print_exc()
    f.close()
# file = "/home/mcesped/projects/ctb-akhanf/cfmm-bids/Khan/epi_iEEG/ieeg/bids/sub-002/ses-002/ieeg/sub-002_ses-002_task-full_run-01_ieeg.edf"
