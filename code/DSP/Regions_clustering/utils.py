import pyedflib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from multiprocessing.pool import Pool
from functools import partial
import scipy
import scipy.io as sio
import scipy.fftpack
import scipy.signal
import matplotlib.pyplot as plt
import re
import plotly.express as px
import os
from bids import BIDSLayout


def welchMethod(data, srate):
    # create Hann window
    win_seconds = 2.0
    winsize = int( win_seconds*srate ) # 2-second window
    hannw = .5 - np.cos(2*np.pi*np.linspace(0,1,winsize))/2

    # number of FFT points (frequency resolution)
    spectres = 0.5; # Hz
    nfft = int(srate/spectres)
    # print('hihi')
    # Apply Welch method
    f, welchpow = scipy.signal.welch(data,fs=srate,window=hannw,
                                    nperseg=winsize,noverlap=winsize/2,nfft=nfft, scaling='density')
    # print(welchpow.shape)
    # Normalizing
    if welchpow.ndim > 1:
        welchpow = np.divide(welchpow, np.sqrt(np.sum(welchpow**2, axis=1)).reshape(welchpow.shape[0],1))
    else:
        welchpow = np.divide(welchpow, np.sqrt(np.sum(welchpow**2)))
    
    # Crop the signal
    min_freq = 1/win_seconds
    max_freq = 80 
    min_id = np.argmin(np.abs(f-min_freq))
    max_id = np.argmin(np.abs(f-max_freq))
    
    return f[min_id:max_id], welchpow[min_id:max_id]

def extract_channel_data(chn_label, edf_file):
    edf_in = pyedflib.EdfReader(edf_file)
    # Get labels from original edf file
    channels_labels = edf_in.getSignalLabels()
    # Get channel id
    chn_id = channels_labels.index(chn_label)
    # Get the data
    signal = edf_in.readSignal(chn_id)
    edf_in.close()
    f, psd = welchMethod(signal, 200) #TODO: change to srate
    return signal, f, psd, chn_label

def plotAllChannels_plotly(freq, welchpow, identifier, out_path=None, show_fig = False, fig=None):
    if fig == None:
        fig = go.Figure()
    for idx in range(welchpow.shape[0]):
        fig.add_trace(go.Scatter(x=freq[idx,:],y=welchpow[idx,:], name=identifier[idx]))
    # Freq bands
    fig.add_trace(go.Scatter(x=[4, 4], y=[-0.1, np.max(welchpow)+0.1], showlegend=False,
                             line=dict(color='black', dash='dash')))
    fig.add_trace(go.Scatter(x=[8, 8], y=[-0.1, np.max(welchpow)+0.1], showlegend=False,
                             line=dict(color='black', dash='dash')))
    fig.add_trace(go.Scatter(x=[13, 13], y=[-0.1, np.max(welchpow)+0.1], showlegend=False,
                             line=dict(color='black', dash='dash')))
    fig.add_trace(go.Scatter(x=[30, 30], y=[-0.1, np.max(welchpow)+0.1], showlegend=False,
                             line=dict(color='black', dash='dash')))
    # Add labels
    fig.add_annotation(y=0.9, x=np.log10(1.5), text=r'$\delta $',showarrow=False, font=dict(size=20))
    fig.add_annotation(y=0.9, x=np.log10(6), text=r'$\theta $',showarrow=False, font=dict(size=20))
    fig.add_annotation(y=0.9, x=np.log10(10.5), text=r'$\alpha $',showarrow=False, font=dict(size=20))
    fig.add_annotation(y=0.9, x=np.log10(21), text=r'$\beta $',showarrow=False, font=dict(size=20))
    fig.add_annotation(y=0.9, x=np.log10(52.5), text=r'$\gamma $',showarrow=False, font=dict(size=20))
    fig.update_xaxes(type="log")
    fig.update_yaxes(range=[0, np.max(welchpow)+0.05])
    fig.update_layout(
    xaxis_title='Frequency [Hz]',
    yaxis_title='Normalized PSD')
    if out_path != None:
        fig.write_html(out_path, include_mathjax='cdn')
    if show_fig:
        fig.show()
    else: 
        fig.data = []

def plotAllChannels_time(time, signal, identifier, flipped=True, out_path=None, show_fig = False, fig=None):
    layout = go.Layout(
        xaxis=dict(
            rangeslider=dict(
                visible = True,
                range=[0, np.max(time)]
            ),
            type='linear'
        )
    )
    if fig == None:
        fig = go.Figure(layout=layout)
    visible = True
    for idx in range(signal.shape[0]):
        fig.add_trace(go.Scatter(x=time,y=signal[idx,:], name=identifier[idx], visible=visible))
        visible = 'legendonly'
    
    fig.update_layout(
    xaxis_title='Time [s]',
    yaxis_title='Amplitude [uV]')
    # Add dropdown
    fig.update_layout(
        updatemenus=[
            dict(
                type = "buttons",
                direction = "left",
                buttons=list([
                    dict(
                        args=["xaxis.range", (0,5)],
                        label="5s",
                        method="relayout"
                    ),
                    dict(
                        args=["xaxis.range", (0,10)],
                        label="10s",
                        method="relayout"
                    ),
                    dict(
                        args=["xaxis.range", (0,np.max(time))],
                        label="all",
                        method="relayout"
                    )
                ]),
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.05,
                xanchor="left",
                y=1.14,
                yanchor="top"
            ),
        ]
    )

    # Add annotation
    fig.update_layout(
        annotations=[
            dict(text="Time range:", showarrow=False,
                                 x=0, y=1.08, yref="paper", align="left")
        ]
    )
    if flipped:
        fig.update_yaxes(autorange="reversed")
    if out_path != None:
        fig.write_html(out_path, include_mathjax='cdn')
    if show_fig:
        fig.show()
    else: 
        fig.data = []

def extract_info(tsv_path, edf_files, edf_images):
    # Load tsv layout
    locations_tsv_layout = BIDSLayout(tsv_path, validate = False)
    # Loop over files and extract info
    psd_regions = {}
    f_regions = {}
    signal_regions = {}
    dict_to_edf = {
        'edf': [],
        'chn': [],
        'region': []
    }
    for edf_image, edf in zip(edf_images, edf_files):
        # Get entities from edf_image to find the tsv
        input_filters = edf_image.get_entities()
        # print(input_filters)
        # print(locations_tsv_layout.get(extension='tsv'))
        # Change a few wc
        input_filters['extension'] = 'tsv'
        input_filters['suffix'] = 'space'
        tsv = locations_tsv_layout.get(**input_filters, return_type='filename')[0]
        # Read location tsv file
        df_locations = pd.read_csv(tsv, sep='\t')
        # Add new column with the regions
        # df_locations['Region']=df_locations['region name'].str.replace('Left-', '')
        # df_locations['Region']=df_locations['Region'].str.replace('Right-', '')
        # Filter white-matter and unknown 
        non_white_matter_unknown_bool = df_locations['region name'].str.contains('White-Matter|Unknown', case=False, regex=True)==False
        filtered_df = df_locations.loc[non_white_matter_unknown_bool]
        # Get unique regions
        # unique_labels = set(filtered_df['Region'])
        unique_labels = set(filtered_df['region name'])
        # unique_labels = ['Amygdala'] # REMOVE
        # Get regions info in dict
        for label in unique_labels:
            # Get channel labels associated to region label
            chn_list = filtered_df.loc[filtered_df['region name']==label,'label'].values.tolist()
            # Get data from edf file
            # ctx = get_context('spawn')
            with Pool(processes=4) as pool:
                # signals = pool.map(partial(extract_channel_data, edf_file=edf), chn_list)
                signals, f, psd, chns = zip(*pool.map(partial(extract_channel_data, edf_file=edf), chn_list))
            # Append to dictionary
            dict_to_edf['edf'] += [edf for curve in psd]
            dict_to_edf['chn'] += chns
            dict_to_edf['region'] += [label for curve in psd]
            if label in psd_regions.keys():
                psd_regions[label] += psd
                f_regions[label] += f
                signal_regions[label] += signals
            else:
                psd_regions[label] = psd
                f_regions[label] = f
                signal_regions[label] = signals
    
    return signal_regions, psd_regions, f_regions, dict_to_edf

# Create new dataframe per region
def df_per_region(info_df, region):
    # Create new dataframe
    region_df = info_df.loc[info_df['region']==region]
    new_id = []
    old_index = []
    subj_list = []
    pattern = r'sub-(.{1,3})_ses-.{1,3}'
    pattern_clip = r'clip-(\d{1,3})'
    for idx in region_df.index:
        chn = region_df.loc[idx,'chn']
        edf = region_df.loc[idx,'edf']
        sub_ses = re.search(pattern, edf).group(0)
        subj = re.search(pattern, edf).group(1)
        clip = re.search(pattern_clip, edf).group(1)
        new_id.append(sub_ses+'_'+chn+'_'+clip)
        old_index.append(idx)
        subj_list.append(subj)
    # Create dataframe 
    new_df = pd.DataFrame({'old_index':old_index,'ID':new_id, 'subj':subj_list})
    return new_df

def errorModifiedZScore(median, MAD, y_pred):
    # Center signal
    y_centered = np.subtract(y_pred, median)
    N = len(y_pred)
    return np.sum(np.divide(y_centered, MAD, out=np.zeros_like(y_centered), where=MAD!=0))/N

def plotPaperFigures(freq, welchpow, out_path=None, output=False, show_fig='Close', ax=None):
    # welchpow: n_chans x n_samples
    # Get median:
    median_welchpow = np.median(welchpow, axis=0)
    # Get std
    n_samples = welchpow.shape[1]
    std = np.zeros(n_samples)
    mean = np.zeros(n_samples)
    for i in range(n_samples):
        (mean[i], std[i]) = scipy.stats.norm.fit(welchpow[:,i].squeeze())
    # Get quartiles
    quant = np.quantile(welchpow, [0.25, 0.75], axis=0)
    # Get max and min
    max_pow = np.max(welchpow, axis=0)
    min_pow = np.min(welchpow, axis=0)

    # Plot
    x_val = [0.5, 4, 8, 13, 30, 80]
    if ax == None:
        fig, ax = plt.subplots()
    ax.semilogx(freq, median_welchpow, 'r')
    ax.fill_between(freq,
                    quant[0,:],
                    quant[1, :],
                    alpha=0.2, color='tab:pink')
    ax.semilogx(freq, max_pow, '--', color='tab:orange')
    ax.semilogx(freq, min_pow, '--', color='tab:orange')
    ax.semilogx([4, 4], [0, 1.1], '--k')
    ax.semilogx([8, 8], [0, 1.1], '--k')
    ax.semilogx([13, 13], [0, 1.1], '--k')
    ax.semilogx([30, 30], [0, 1.1], '--k')
    ax.set_xticks(x_val)
    ax.set_xticklabels(x_val)
    ax.set_xlim([0.5,80])
    ax.set_ylim([0, 1])
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Normalized PSD')
    # ax.set_xscale('log')
    if output:
        fig.savefig(out_path)
    if show_fig=='Close':
        plt.close()
    elif show_fig == True:
        plt.show()
    return median_welchpow, mean, std

def plotAllChannels(freq, welchpow, out_path=None, output=False, show_fig = False, ax=None):
    x_val = [0.5, 4, 8, 13, 30, 80]
    if ax == None:
        fig, ax = plt.subplots()
    ax.semilogx(freq,welchpow)
    ax.semilogx([4, 4], [0, 1.1], '--k')
    ax.semilogx([8, 8], [0, 1.1], '--k')
    ax.semilogx([13, 13], [0, 1.1], '--k')
    ax.semilogx([30, 30], [0, 1.1], '--k')
    ax.text(5.5, 0.9, r'$\theta $')
    ax.text(9.5, 0.9, r'$\alpha $')
    ax.text(18, 0.9, r'$\beta $')
    ax.text(45, 0.9, r'$\gamma $')
    ax.set_xticks(x_val)
    ax.set_xticklabels(x_val)
    ax.set_xlim([0.5,80])
    ax.set_ylim([0, 1])
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Normalized PSD')
    if output:
        fig.savefig(out_path)
    if show_fig:
        plt.show()
    elif show_fig == 'Close':
        plt.close()

def plot_multiple_dist(df_plot, labels, psd, f, clustering_alg, outpath):
    num_clusters = len(labels)
    psd_dict = {}
    comp_idx_dict = {}
    fig, axs = plt.subplots(nrows=num_clusters, ncols=1, figsize=(5,4*num_clusters))
    for idx,label in enumerate(labels):
        comp_idx_dict[label] = df_plot.loc[df_plot[clustering_alg]==label, ['1 component', '2 component', '3 component']].index
        psd_dict[label] = psd[comp_idx_dict[label], :]
        plotPaperFigures(f[0], psd_dict[label], show_fig=False, ax=axs[idx])
        axs[idx].set_title(f'Cluster label {label}')
    plt.tight_layout()
    fig.savefig(outpath)
    plt.close()
    return psd_dict, comp_idx_dict

def plot_multiple_allchannels(psd_dict, comp_idx_dict, f, outpath):
    labels = list(psd_dict.keys())
    num_clusters = len(labels)
    fig, axs = plt.subplots(nrows=num_clusters, ncols=1, figsize=(5,4*num_clusters))
    for idx, label in enumerate(labels):
        plotAllChannels(f[comp_idx_dict[label], :].T, psd_dict[label].T, show_fig=False, ax=axs[idx])
        axs[idx].set_title(f'Cluster {label}')
    plt.tight_layout()
    fig.savefig(outpath)
    plt.close()

def save_results_clustering(clustering_type, out_path_region, df_plot, psd, f):
    # Plot and save scatterplot
    fig2 = px.scatter_3d(df_plot, x='1 component', y='2 component', z='3 component', hover_name='ID', color = clustering_type)
    fig2.write_html(os.path.join(out_path_region, f'{clustering_type}_scatterplot.html'), include_mathjax='cdn')
    del fig2
    # Plot distribution of PSDs
    labels = df_plot[clustering_type].unique().tolist()
    psd_dict, comp_idx_dict = plot_multiple_dist(df_plot, labels, psd, f, clustering_type, os.path.join(out_path_region, f'{clustering_type}_PSD_distribution.png'))
    # Plot all channels
    plot_multiple_allchannels(psd_dict, comp_idx_dict, f, os.path.join(out_path_region, f'{clustering_type}_PSDs_per_cluster.png'))

    # Criteria based on number of subjects
    n_subj_total = df_plot['subj'].unique().size
    keep = []
    labels_count = {}
    for label in labels:
        n_subj_label = len(df_plot.loc[df_plot[clustering_type]==label]['subj'].unique())
        labels_count[label] = n_subj_label
        if n_subj_label >= 0.7*n_subj_total:
            keep.append(label)
    # Write txt with the info
    with open(os.path.join(out_path_region, f'{clustering_type}_subj_criteria.txt'), 'w') as f:
        f.write(f'The clusters that should be kept are: {str(keep)}\n')
        f.write(f'The total number of subjects are {n_subj_total}\n')
        f.write('The number of subjects per cluster are:\n')
        for key in labels_count:
            f.write(f'Cluster {key}: {labels_count[key]}\n')