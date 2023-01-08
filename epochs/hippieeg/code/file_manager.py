import pyedflib
import numpy as np
import os, fnmatch
from collections import OrderedDict
from pathlib import Path
from typing import Union
from functools import partial
import traceback
from multiprocessing.pool import Pool
import re
import pandas as pd
import nibabel as nb
import mne


# Function to look for timestamps
def extract_time_ids(epoch_id, time_vector, timestamps_array, srate):
    temp = np.asfortranarray(np.subtract(time_vector,timestamps_array[epoch_id]))
    t_init_id = np.abs(temp).argmin() ## CANNOT BE ZERO, change to a not magic 
    t_end_id = int(np.floor(t_init_id+240*srate+1)) # 4 min = 240 s
    return (t_init_id, t_end_id)

# Function to create bipolar channels from given unipolars
def create_bipolar_comb(id, dict_key, channels_dict):
    bipolar_chn = dict_key+channels_dict[dict_key][id]+'-'+channels_dict[dict_key][id+1] 
    chn1_label = dict_key+channels_dict[dict_key][id]
    chn2_label = dict_key+channels_dict[dict_key][id+1]
    return (bipolar_chn, chn1_label, chn2_label)

# Function to extract position of bipolar channels
def bipolar_info(id, dict_key, channels_dict, elec_pos):
    # Bipolar channel label
    bipolar_chn = dict_key+channels_dict[dict_key][id]+'-'+channels_dict[dict_key][id+1] 
    # Channels' labels
    chn1_label = dict_key+channels_dict[dict_key][id]
    chn2_label = dict_key+channels_dict[dict_key][id+1]
    # Extract positions
    inf_chn1 = elec_pos.loc[elec_pos.label == chn1_label]
    inf_chn2 = elec_pos.loc[elec_pos.label == chn2_label]
    # print(inf_chn1)
    data = {
        'type': inf_chn1.type.values[0],
        'group': inf_chn1.orig_group.values[0],
        'label': bipolar_chn,
        'x': (inf_chn1.x.values[0] + inf_chn2.x.values[0])/2,
        'y': (inf_chn1.y.values[0] + inf_chn2.y.values[0])/2,
        'z': (inf_chn1.z.values[0] + inf_chn2.z.values[0])/2
    }
    return data
    

# Function to create a bipolar channel list from 
def create_bipolars(electrodes_df, processes):
    channels = dict((label,[]) for label in electrodes_df.orig_group.unique())
    pattern = r'([A-Z]+)(\d+)'
    electrode_labels = electrodes_df.label.values
    # Extract channels info
    for electrode in electrodes_df.label.values:
        match = re.match(pattern, electrode, re.IGNORECASE)
        channels[match.group(1)].append(match.group(2))
    # print(channels)
    # Create new list
    bipolar_list = []
    bipolar_info_dicts = []
    for key in  channels.keys():
        # Extract bipolar list (original channels + bipolar channel)
        with Pool(processes=processes) as pool:
                bipolar_list = bipolar_list + pool.map(partial(create_bipolar_comb, dict_key=key, channels_dict=channels), 
                                        list(range(len(channels[key])-1)))
        # Extract bipolar channels information
        with Pool(processes=processes) as pool:
                bipolar_info_dicts = bipolar_info_dicts + pool.map(partial(bipolar_info, dict_key=key, channels_dict=channels,
                                                          elec_pos=electrodes_df), list(range(len(channels[key])-1)))
        bipolar_elec = pd.DataFrame(columns=np.concatenate((electrodes_df.columns[0:5],['group'])))
        data = pd.DataFrame(bipolar_info_dicts)
        bipolar_elec = pd.concat([bipolar_elec, data], ignore_index=True)
    return bipolar_list, bipolar_elec

# Function to extract info from each channel
def extract_channel_data(chn_number, edf_file, srate_data, time_ids, bipolar_list):
    edf_in = pyedflib.EdfReader(edf_file)
    # Get labels from original edf file
    channels_labels = edf_in.getSignalLabels()
    # Get indexes of channels
    print('new iter')
    chn1_id = channels_labels.index(bipolar_list[chn_number][1])
    chn2_id = channels_labels.index(bipolar_list[chn_number][2])
    print(chn_number)
    chn_data = np.array([], dtype=float)
    for t_id in time_ids:
        n_val = t_id[1]-t_id[0]
        signal_chn1 = edf_in.readSignal(chn1_id, start=t_id[0], n=n_val)
        signal_chn2 = edf_in.readSignal(chn2_id, start=t_id[0], n=n_val)
        signal_bipolar = signal_chn1 - signal_chn2
        chn_data = np.hstack([chn_data, 
                            signal_bipolar, 
                            np.zeros(int(60*srate_data))])
    # Deallocate space in memory
    edf_in.close()
    del signal_chn1
    del signal_chn2
    return chn_data

# Function to extract headers for bipolar channels
def extract_channel_header(chn_number, original_headers, bipolar_list, channels_labels):
    # Get indexes of channels
    chn1_id = channels_labels.index(bipolar_list[chn_number][1])
    # Update header
    chn_header = original_headers[chn1_id]
    chn_header['label'] = bipolar_list[chn_number][0]
    return chn_header

# Function to get label map
def get_colors_labels():
    with open('/home/mcesped/scratch/HippiEEGAtlas/FreeSurferColorLUT.txt', 'r') as f:
        raw_lut = f.readlines()

    # read and process line by line
    label_map = pd.DataFrame(columns=['Label', 'R', 'G', 'B'])
    for line in raw_lut:
        # Remove empty spaces
        line = line.strip()
        if not (line.startswith('#') or not line):
            s = line.split()
            # info = list(filter(None, info))
            id = int(s[0])
            info_s = {
                'Label': s[1],
                'R': int(s[2]),
                'G': int(s[3]),
                'B': int(s[4])
            }
            # info_s['A'] = 0 if (info_s['R']==0 & info_s['G']==0 & info_s['B']==0) else 255
            info_s = pd.DataFrame(info_s, index=[id])
            label_map = pd.concat([label_map,info_s], axis=0)
        label_map[['R','G','B']] = label_map[['R','G','B']].astype('int64')
    return label_map

# Function to get label id based on parcellation obj
def get_electrodes_id(parc, elec_df, non_cont_to_cont_tf):
    # Load data of parcellations
    data_parc = np.asarray(parc.dataobj)
    # Coordinates in MRI RAS
    mri_ras_mm = elec_df[['x','y','z']].values
    # Transform from contrast mri ras to non-contrast MRI ras
    mri_ras_mm = mne.transforms.apply_trans(non_cont_to_cont_tf, mri_ras_mm)
    # To voxels
    inv_affine = np.linalg.inv(parc.affine)
    # here's where the interpolation should be performed!!
    vox = np.round(mne.transforms.apply_trans(inv_affine, mri_ras_mm)).astype(int)
    id = data_parc[vox[:,0], vox[:,1], vox[:,2]]
    return id

# Function to get rgb values for each contact
def get_label_rgb(parc, elec_df, non_cont_to_cont_tf, label_map):
    # vox, data_parc = ras2vox(parc, elec_df, non_cont_to_cont_tf)
    id = get_electrodes_id(parc, elec_df, non_cont_to_cont_tf)
    vals = label_map.loc[id, ['Label','R', 'G', 'B']].to_numpy()
    vals = np.c_[id,vals]
    vals = pd.DataFrame(data=vals, columns=['Label ID','Label','R', 'G', 'B'])
    vals = pd.concat([elec_df, vals], axis=1)
    return vals

# Function to read matrix
def readRegMatrix(trsfPath):
	with open(trsfPath) as (f):
		return np.loadtxt(f.readlines())

# Function to create tsv with bipolar channels info
def create_bipolar_tsv(parc_path, noncon_to_con_tf_path, bipolar_info, out_tsv_name):
    # Only run if not generated previously
    if not os.path.exists(out_tsv_name):
        # Load labels from LUT file
        labels = get_colors_labels()
        # Load parcellation file
        parc_obj = nb.load(parc_path)
        data_parc = np.asarray(parc_obj.dataobj)
        # The transform file goes from contrast to non-contrast. The tfm, when loaded in slicer actually goes
        # from non-contrast to contrast but the txt is inversed!
        t1_transform=readRegMatrix(noncon_to_con_tf_path)
        # Create df
        df = get_label_rgb(parc_obj, bipolar_info, t1_transform, labels)
        df.to_csv(out_tsv_name, sep = '\t')

# Function to create EDF file with bipolar data
def create_EDF(edf_file, time_stamps, bipolar_channels, out_path, processes):
    try:
        edf_in = pyedflib.EdfReader(edf_file)
        # First import labels
        labels = edf_in.getSignalLabels()
        # Create file:
        edf_out = pyedflib.EdfWriter(out_path, len(bipolar_channels), file_type=pyedflib.FILETYPE_EDFPLUS)
        # First set the data from the header of the edf file:
        edf_out.setHeader(edf_in.getHeader())
        headers_orig = edf_in.getSignalHeaders()
        # f.datarecord_duration gives the value is sec and setDatarecordDuration receives it in units
        # of 10 ms. Therefore: setDatarecordDuration = datarecord_duration*10^6 / 10
        edf_out.setDatarecordDuration(int(edf_in.datarecord_duration*100000)) # This actually is used to set the sample frequency
        # Set each channel info:
        # Sampling rate:
        srate = edf_in.getSampleFrequencies()[0]/edf_in.datarecord_duration
        # Build epochs
        N = edf_in.getNSamples()[0]
        # Time vector:
        t = np.arange(0, N)/srate
        # Relative initial time for epochs
        t_0 = t[np.abs(np.subtract(t,time_stamps[0])).argmin()]
        edf_out.writeAnnotation(0, -1, "Recording starts")
        # Close file
        edf_in.close()
        # Create time ids
        list_epochs_ids = list(range(time_stamps.size))
        print('Time part')
        # Extracting time indexes based on time stamps
        with Pool(processes=processes) as pool:
            t_ids = pool.map(partial(extract_time_ids, time_vector=t, timestamps_array=time_stamps, srate=srate), 
                            list_epochs_ids)
        # Write annotations at the different epochs times (beginning and end of epochs)
        for id, (t_init_id, t_end_id) in enumerate(t_ids):
            # Write annotations
            edf_out.writeAnnotation(t[t_init_id]-t_0, -1, f"Epoch #{id+1} starts.")
            edf_out.writeAnnotation(t[t_end_id]-t_0, -1, f"Epoch #{id+1} ends.")
        # Deallocate space in memory
        del t
        # Extract channel information:
        chn_lists = range(len(bipolar_channels))
        print('Channel part')
        # Create bipolar signals:
        with Pool(processes=processes) as pool2:
            channel_data = pool2.map(partial(extract_channel_data, edf_file=edf_file, 
                                            srate_data=srate, time_ids=t_ids,
                                            bipolar_list=bipolar_channels), chn_lists)
        # Create headers:
        with Pool(processes=processes) as pool2:
            headers = pool2.map(partial(extract_channel_header, 
                                      original_headers=headers_orig,
                                      bipolar_list=bipolar_channels,
                                      channels_labels=labels), chn_lists)
        # Edit headers to make them compliant with edf files
        for header in headers:
            header['physical_max'] = int(header['physical_max'])
            header['physical_min'] = int(header['physical_min'])
            if len(str(header['physical_max']))>8:
                header['physical_max'] = int(str(header['physical_max'])[0:8])
            if len(str(header['physical_min']))>8:
                header['physical_min'] = int(str(header['physical_min'])[0:8])
        
        edf_out.setSignalHeaders(headers)
        edf_out.writeSamples(channel_data)
        edf_out.close()
    except Exception:
        traceback.print_exc()
        edf_out.close()
        edf_in.close()

def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result

def bids_folder(
    root: Union[str, Path] = None,
    datatype: str = None,
    prefix: str = None,
    suffix: str = None,
    subject: str = None,
    session: str = None,
    include_subject_dir: bool = True,
    include_session_dir: bool = True,
    **entities: str,
):
    # Recovered from snakebids code
    # replace underscores in keys (needed so that users can use reserved
    # keywords by appending a _)
    entities = {k.replace("_", ""): v for k, v in entities.items()}

    # strict ordering of bids entities is specified here:
    # pylint: disable=unsubscriptable-object
    order: OrderedDict[str, Optional[str]] = OrderedDict(
        [
            ("task", None),
            ("acq", None),
            ("ce", None),
            ("rec", None),
            ("dir", None),
            ("run", None),
            ("mod", None),
            ("echo", None),
            ("hemi", None),
            ("space", None),
            ("res", None),
            ("den", None),
            ("label", None),
            ("desc", None),
        ]
    )

    # Now add in entities (this preserves ordering above)
    for key, val in entities.items():
        order[key] = val
    
    # Form folder using list similar to filename, above. Filter out Nones, and convert
    # to Path.
    folder = Path(
        *filter(
            None,
            [
                str(root) if root else None,
                f"sub-{subject}" if subject and include_subject_dir else None,
                #f"ses-{session}" if session and include_session_dir else None,
                #datatype,
            ],
        )
    )

    return str(folder)