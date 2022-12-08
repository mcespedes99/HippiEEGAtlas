import pyedflib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import traceback
from create_edf import create_EDF
import argparse

parser = argparse.ArgumentParser(description='Script to extract epochs\
                                             from a *.edf file.')
parser.add_argument("I", metavar='INPUT',
                    help='Path to the input file with extension *.edf')
parser.add_argument("O", metavar='OUTPUT',
                    help='Path to the output folder.')

args = parser.parse_args()

input_path = args.I
output_path = args.O

# Import file:
try:
    f = pyedflib.EdfReader(input_path)
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
        # Here call function
        create_EDF(f, time_stamps, output_path)
except FileNotFoundError:
    raise FileNotFoundError('The *.edf file was not found in the indicated \
        path.')
except BaseException:
    traceback.print_exc()
# file = "/home/mcesped/projects/ctb-akhanf/cfmm-bids/Khan/epi_iEEG/ieeg/bids/sub-002/ses-002/ieeg/sub-002_ses-002_task-full_run-01_ieeg.edf"