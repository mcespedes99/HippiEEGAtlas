"""
Utility functions for iEEGPrep.
Author: Mauricio Cespedes Tenorio
"""

def get_chn_positions(chn_csv_path, trsfPath=None):
    import numpy as np
    import pandas as pd
    """Creates dictionary with the position of each electrode.
    Parameters
    ----------
    ch_csv_path : str
        Path to csv containing electrodes positions.

    Returns : dictionary with position (x,y,z) of each electrode.
    -------
        
    """
    elec_pos = pd.read_csv(chn_csv_path)
    chn_pos = {}
    for i in np.arange(len(elec_pos)):
        label = elec_pos.loc[[i], ['label']].values[0][0]
        pos = elec_pos.loc[[i], ['x','y','z']].values[0]/1000
        if trsfPath != None:
            tfm = readRegMatrix(trsfPath)
            pos = contrast_to_non_contrast(pos, tfm)
        pos = pos.tolist()
        chn_pos[label] = pos
    return chn_pos

def get_montage(ch_pos, subject, subjects_dir):
    import mne
    """Get montage given Surface RAS (aka mri coordinates in MNE)
    Recovered from:
    https://mne.discourse.group/t/question-re-seeg-source-localization-using-mne/4411/5
    Parameters
    ----------
    ch_pos : dict
        Dictionary of channel positions. Keys are channel names and values
        are 3D coordinates - array of shape (3,) - in native digitizer space
        in m.
    subject ： str
        The name of subject in FreeSurfer
    subjects_dir : str
        The directory of your FreeSurfer subject directory
    contrast : bool
        Boolean indicating whether it is required to transform the coordinates from 
        contrast to non-contrast space.

    Returns : head montage
    -------
        
    """
    subj_trans = mne.coreg.estimate_head_mri_t(subject, subjects_dir)
    mri_to_head_trans = mne.transforms.invert_transform(subj_trans)
    print('Start transforming mri to head')
    print(mri_to_head_trans)

    montage_mri = mne.channels.make_dig_montage(ch_pos, coord_frame='mri')
    montage = montage_mri.copy()
    montage.add_estimated_fiducials(subject, subjects_dir)
    montage.apply_trans(mri_to_head_trans)
    return montage

def readRegMatrix(trsfPath):
    import mne
    """Reads transformation matrix.
    Parameters
    ----------
    trsfPath : str
            Path to transform from contrast RAS to non-constrast RAS.

    Returns : Transform from contrast RAS to non-contrast RAS space.
    -------
        
    """
    import numpy as np

    with open(trsfPath) as (f):
        return np.loadtxt(f.readlines())

def contrast_to_non_contrast(pnt, tfm):
    import mne
    """Transforms electrode coordinate from contrast space to non-contrast space.
    Parameters
    ----------
    pnt : ndarray (3,)
        Coordinate of the electrode to transform, in mm and RAS space
        from the contrast MRI.
    tfm : ndarray
        Transform from contrast RAS to non-contrast RAS space.

    Returns : Coordinates in RAS for non-constrast MRI space.
    -------
        
    """
    mri_ras_mm = mne.transforms.apply_trans(tfm, pnt)
    return mri_ras_mm

def get_orig_data(epoch_path, chn_pos):
    """Get labels and start-end timestamps for each epoch.
    """
    import pyedflib
    import pandas as pd
    import numpy as np
    # Read data
    edf_in = pyedflib.EdfReader(epoch_path)
    # Read labels
    labels = edf_in.getSignalLabels()
    # Get annotations from edf
    annot = edf_in.readAnnotations()
    annot = {
        'Onset': annot[0],
        'Duration': annot[1],
        'event': annot[2]
    }
    annot = pd.DataFrame(annot)
    
    # Get start and end times
    start_times = annot.Onset.to_numpy()[annot['event'].str.match(r'Epoch #\d starts.')]
    end_times = annot.Onset.to_numpy()[annot['event'].str.match(r'Epoch #\d ends.')]
    # Concatenate two epochs
    timestamps_epochs = np.c_[start_times, end_times]
    edf_in.close()
    
    return labels, timestamps_epochs

def segment_signal(signal, srate, time_epoch=5):
    import numpy as np
    n_epoch = int(time_epoch*srate) # 5 seconds by default
    # Initialize segmented signal
    signal_epoch = np.zeros((int(signal.shape[1]/n_epoch), signal.shape[0], n_epoch))
    id = 0
    start_id = []
    end_id = []
    # Segment the signal
    for epoch_id in np.arange(int(signal.shape[1]/n_epoch)):
        tmp = signal[:,id:id+n_epoch]
        start_id.append(id)
        end_id.append(id+n_epoch)
        signal_epoch[epoch_id,:,:] = tmp
        id += n_epoch
    epochs_ids = {
        'Start ID': start_id,
        'End ID': end_id
        }
    return signal_epoch, epochs_ids

def create_mne_epochs(signal, chn_labels, srate, montage, time_epoch=5):
    import numpy as np
    import mne
    # Divide the signal into small epochs
    signal_epoch, epochs_ids = segment_signal(signal, srate, time_epoch)
    # Create information for MNE structure
    info = mne.create_info(ch_names=chn_labels,
                        ch_types=['seeg'] * len(chn_labels),
                        sfreq=srate)
    # Create MNE epoch array 
    mne_epochs = mne.EpochsArray(signal_epoch, info)
    # Set montage
    mne_epochs.set_montage(montage)
    return mne_epochs, epochs_ids
        
def run_autoreject(mne_epoch_array, exclude = []):
    from autoreject.autoreject import AutoReject, compute_thresholds
    import mne
    import numpy as np
    # Create Autoreject instance
    ar = AutoReject(random_state=42, n_jobs=-1, verbose=True)
    # Run autoreject
    epochs_ar, reject_log = ar.fit_transform(mne_epoch_array, return_log=True)
    # Create 'clean'/'noisy' labels (n_epochs x n_channels)
    noise_labels = np.copy(reject_log.labels).astype('object')
    # Manage possible nan values 
    np.nan_to_num(noise_labels, copy=False)
    # Include bad epochs 
    noise_labels[reject_log.bad_epochs] = 1
    # Define dict for possible values
    noise_map = {0: 'C', 1: 'N', 2: 'N'} #(0.6, 0.6, 0.6, 1.0)
    for key in list(noise_map.keys()):
        noise_labels[noise_labels==key] = noise_map[key]
    return epochs_ar, noise_labels
    

def clean_signal(edf_path, chn_csv_path, subject, subjects_dir, trsfPath=None, time_epoch=5):
    import pyedflib
    import numpy as np
    import pandas as pd
    # Begin by getting the position of the electrodes in RAS space
    chn_pos = get_chn_positions(chn_csv_path, trsfPath)
    # Extract the labels and timestamps required
    labels, timestamps_epochs = get_orig_data(edf_path, chn_pos)
    # Defining length of epochs based on first epoch
    t_init = timestamps_epochs[0,0]
    t_end = timestamps_epochs[0,1]
    length_epoch = t_end-t_init
    # Open edf file
    edf_in = pyedflib.EdfReader(edf_path)
    # Channels to extract
    keys = list(chn_pos.keys())
    # Sample rate
    srate = edf_in.getSampleFrequencies()[0]/edf_in.datarecord_duration
    # Number of samples
    N=edf_in.getNSamples()[0]
    # Create time vector using srate
    t = np.arange(0, N)/srate
    # Create sEEG montage
    montage = get_montage(chn_pos, subject, subjects_dir)
    # Initiate clean signal
    clean = np.array([]).reshape(len(keys),0)
    # Initiate csv epoch file
    cols = ['Epoch #', 'Start ID', 'End ID']+keys
    df_epochs = pd.DataFrame(columns=cols)
    # Last epoch number
    last_epoch = 0
    # Run the algorithm per epoch
    n_epochs = timestamps_epochs.shape[0]
    for epoch_id in np.arange(n_epochs):
        t_init = timestamps_epochs[epoch_id,0]
        t_init_next = np.max(timestamps_epochs[epoch_id,0], t_init+length_epoch)
        # Find idx for t_init
        t_init_id = np.argmin((np.abs(t-t_init)))
        # Create signal for that epoch
        signal = np.array([], dtype=np.int64).reshape(0,length_epoch)
        signal_not_clean = np.array([], dtype=np.int64).reshape(0,length_epoch)
        # Extract signal per channel
        for chan in keys:
            id_ch = labels.index(chan)
            chn_sig = edf_in.readSignal(id_ch, start = t_init_id, n = t_init_next)
            chn_sig_epoch = chn_sig[0:length_epoch]
            signal = np.vstack([signal, chn_sig_epoch])
            signal_not_clean = np.vstack([signal_not_clean, chn_sig[length_epoch:]])
        # Create MNE epochs
        mne_epochs, epochs_ids = create_mne_epochs(signal, keys, srate, montage, time_epoch)
        # Update IDs
        start_IDs = epochs_ids['Start ID']+t_init_id
        end_IDs = epochs_ids['End ID']+t_init_id
        # Epochs #s
        epoch_num = np.arange(last_epoch, last_epoch+len(start_IDs))
        last_epoch = last_epoch+len(start_IDs)
        # Run autoreject
        epochs_ar, noise_labels = run_autoreject(mne_epochs)
        # Create noise df
        IDs_array = np.array([start_IDs,end_IDs]).T
        noise_array = np.c_[epoch_num, IDs_array, noise_labels]
        df_tmp = pd.DataFrame(data = noise_array, columns = cols)
        df_epochs = pd.concat([df_epochs, df_tmp])

        # Reshape to n_chn x n_time
        clean_sig = epochs_ar.get_data()
        clean_sig = clean_sig.swapaxes(0,1).reshape(2,-1)
        # Attach the non-clean part of the signal
        clean_sig = np.hstack([clean_sig, signal_not_clean])
        # Update clean signal
        clean = np.hstack([clean, clean_sig])

        return clean, df_epochs