import pyedflib
import numpy as np

def create_EDF(edf_in, time_stamps, out_path):
    # First import labels
    labels = edf_in.getSignalLabels()
    # Create file:
    edf_out = pyedflib.EdfWriter(out_path, len(labels), file_type=pyedflib.FILETYPE_EDFPLUS)
    try:
        # First set the data from the header of the edf file:
        edf_out.setHeader(edf_in.getHeader())
        # f.datarecord_duration gives the value is sec and setDatarecordDuration receives it in units
        # of 10 ms. Therefore: setDatarecordDuration = datarecord_duration*10^6 / 10
        # int(edf_in.datarecord_duration*(100000)))
        edf_out.setDatarecordDuration(int(edf_in.datarecord_duration*100000)) # This actually is used to set the sample frequency
        # Set each channel info:
        # First create an empty list for the data
        channel_data = [np.array([], dtype=float) for chn in range(len(labels))]
        # Sampling rate:
        srate = edf_in.getSampleFrequencies()[0]/edf_in.datarecord_duration
        # Build epochs
        N = edf_in.getNSamples()[0]
        # Time vector:
        t = np.arange(0, N)/srate
        # Relative initial time for epochs
        t_0 = t[np.abs(np.subtract(t,time_stamps[0])).argmin()]
        edf_out.writeAnnotation(0, -1, "Recording starts")
        # Change this for to 2 separate fors. The first should find and store the t_ids and the second one 
        # should iterate across channels
        t_ids = []
        for epoch_id in range(time_stamps.size):
            temp = np.asfortranarray(np.subtract(t,time_stamps[epoch_id]))
            t_init_id = np.abs(temp).argmin() ## CANNOT BE ZERO, change to a not magic #
            t_end_id = int(np.floor(t_init_id+240*srate+1)) # 4 min = 240 s
            t_ids.append((t_init_id, t_end_id))
            # Write annotations
            edf_out.writeAnnotation(t[t_init_id]-t_0, -1, f"Epoch #{epoch_id+1} starts.")
            edf_out.writeAnnotation(t[t_end_id]-t_0, -1, f"Epoch #{epoch_id+1} ends.")
        # Deallocate space in memory
        del t
        for chn in range(len(labels)):
            signal = edf_in.readSignal(chn)
            for t_id in t_ids:
                channel_data[chn] = np.hstack([channel_data[chn], signal[t_id[0]:t_id[1]], np.zeros(int(60*srate))])
                # Deallocate space in memory
            del signal
        # print(channel_data)
        headers = edf_in.getSignalHeaders()
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