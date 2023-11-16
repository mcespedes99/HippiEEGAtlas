import numpy as np
import xarray as xr
from xarray.core.weighted import DataArrayWeighted
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
import scipy
import scipy.io as sio
import scipy.fftpack
import scipy.signal
import copy
import os
from sklearn.metrics import mean_squared_error
import re
import matplotlib.pyplot as plt


# Computation
# Get bandpower per freq band
def bandpower(Pxx, f, fmin, fmax):
    ind_min = np.argmax(f > fmin) - 1
    ind_max = np.argmax(f > fmax) - 1
    return np.trapz(Pxx[:,ind_min: ind_max], f[ind_min: ind_max], axis=-1)

from sklearn.preprocessing import normalize
def compute_bandpower(dataset: xr.Dataset, group_coord: str) -> xr.Dataset:
    # dataset has to have a 'psd' data_var and a 'frequency' dimension associated with it. Also an 'n' dim
    # will compute for each unique group in group_coord. gr
    # Initialize array based on 'n'
    bandpow = np.zeros((len(dataset['n']), 5)) # 5 frequency bands
    f_bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    # Compute groups
    groups = np.unique(dataset[group_coord])
    for group in groups:
        # Get the data for the group in form of np array
        dataarray = dataset['psd'].where(dataset[group_coord]==group, drop=True)
        psd = dataarray.to_numpy()
        f = dataarray['frequency'].to_numpy()
        # Calculate per freq band
        freq_bands = [0.5,4,8,13,30,80]
        bandpow_bands = np.zeros((psd.shape[0], 5))
        for i in range(len(f_bands)): 
            bandpow_bands[:,i] = bandpower(psd, f, freq_bands[i], freq_bands[i+1])
        # Normalize (used l1 as it's based on Euclidean distance, same as here: https://github.com/jbernabei/iEEG_atlas/blob/main/support_files/create_univariate_atlas.m#L1)
        bandpow_bands = normalize(bandpow_bands, norm='l1', axis=1)
        # Add to final array
        bandpow[dataarray.indexes['n']] = bandpow_bands
    # Convert to DataArray
    # Dimensions
    coords = {
        'f_bands': f_bands,
        'n' : np.arange(bandpow.shape[0])
    }
    # Create dataarray
    array_bands = xr.DataArray(
        bandpow,
        coords=coords,
        dims=["n", "f_bands"]
    )
    # Add to a copy of the original dataset
    # Add to dataset
    ds_return = dataset.copy()
    ds_return['bandpow'] = array_bands
    return ds_return

def psd_xarray(ds: xr.Dataset, srate: float)->xr.Dataset:
    # ds data_vars: time_domain ; dims: n, time
    # Get data
    data_array = ds['time_domain'].to_numpy()
    # Convert
    f, psd = welchMethod(data_array, srate)
    # Include dimensions
    coords = {
        'frequency' : f,
        'n' : np.arange(psd.shape[0])
    }
    # Create dataarray
    ds_psd = xr.DataArray(
        psd,
        coords=coords,
        dims=["n", "frequency"]
    )
    # Add to dataset
    ds_return = ds.copy()
    ds_return['psd'] = ds_psd
    return ds_return

# Plot functions
def plotPaperFigures(welchpow, freq, out_path=None, output=False, show_fig='Close', ax=None, title=None):
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
    default_x_ticks = range(len(x_val))
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
    ax.set_xlabel('frequency [Hz]')
    ax.set_ylabel('Power')
    if title:
        ax.set_title(title)
    # ax.set_xscale('log')
    if output:
        fig.savefig(out_path)
    if show_fig=='Close':
        plt.close()
    elif show_fig == True:
        plt.show()
    return None
    # return median_welchpow, mean, std

def plot_xarrays(welchpow, out_path=None, output=False, show_fig='Close', ax=None, title=None):
    # welchpow: weighted xarray 'n' x 'frequency'
    # Get median:
    median_welchpow = welchpow.quantile(0.5, dim='n').to_numpy()
    # Get quartiles
    quant_25 = welchpow.quantile(0.25, dim='n').to_numpy()
    quant_75 = welchpow.quantile(0.75, dim='n').to_numpy()
    # Get max and min
    max_pow = welchpow.obj.max(dim='n').to_numpy()
    min_pow = welchpow.obj.min(dim='n').to_numpy()
    # Get frequency
    freq = welchpow.obj.coords['frequency'].to_numpy()
    
    # Plot
    x_val = [0.5, 4, 8, 13, 30, 80]
    default_x_ticks = range(len(x_val))
    if ax == None:
        fig, ax = plt.subplots()
    ax.semilogx(freq, median_welchpow, 'r')
    ax.fill_between(freq,
                    quant_25,
                    quant_75,
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
    ax.set_ylabel('Power')
    if title:
        ax.set_title(title)
    # ax.set_xscale('log')
    if output:
        fig.savefig(out_path)
    if show_fig=='Close':
        plt.close()
    elif show_fig == True:
        plt.show()
    return None

import scipy.signal
def welchMethod(data, srate):
    # Data: n_chans x n_samples
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
    print(welchpow.shape)
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
    
    if welchpow.ndim > 1:
        return f[min_id:max_id], welchpow[:,min_id:max_id]
    else:
        return f[min_id:max_id], welchpow[min_id:max_id]
    
import matplotlib.colors as mcolors
def plot_comparisons(welchpow_list, datasets_names, out_path=None, output=False, show_fig='Close', ax=None, title=None):
    # welchpow_list is a list of no more than 10 welchpows, which should be weighted x arrays of dim: 'n' x 'frequency'
    # Plot setup
    x_val = [0.5, 4, 8, 13, 30, 80]
    default_x_ticks = range(len(x_val))
    if ax == None:
        fig, ax = plt.subplots()
    # Colors
    colors = list(mcolors.TABLEAU_COLORS.keys())
    for i in range(len(welchpow_list)):
        welchpow = welchpow_list[i]
        color = colors[i]
        name = datasets_names[i]
        # Get median:
        median_welchpow = welchpow.quantile(0.5, dim='n').to_numpy()
        # Get quartiles
        quant_25 = welchpow.quantile(0.25, dim='n').to_numpy()
        quant_75 = welchpow.quantile(0.75, dim='n').to_numpy()
        # Get frequency
        freq = welchpow.obj.coords['frequency'].to_numpy()

        ax.semilogx(freq, median_welchpow, color, label=f'{name}')
        ax.fill_between(freq,
                        quant_25,
                        quant_75,
                        alpha=0.2, color=color)
    ax.semilogx([4, 4], [0, 1.1], '--k')
    ax.semilogx([8, 8], [0, 1.1], '--k')
    ax.semilogx([13, 13], [0, 1.1], '--k')
    ax.semilogx([30, 30], [0, 1.1], '--k')
    ax.set_xticks(x_val)
    ax.set_xticklabels(x_val)
    ax.set_xlim([0.5,80])
    ax.set_ylim([0, 1])
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Power')
    plt.legend()
    if title:
        ax.set_title(title)
    # ax.set_xscale('log')
    if output:
        fig.savefig(out_path)
    if show_fig=='Close':
        plt.close()
    elif show_fig == True:
        plt.show()

def boxplot_bands_xarray(ds: xr.Dataset, data_var: str, group: str):
    # Convert to dataframe
    data_bands = ds["bandpow"].to_numpy()
    df_bands = pd.DataFrame(data_bands, columns=ds["f_bands"].to_numpy())
    # Add other valuable cols
    cols = [coord for coord in list(ds.coords.keys()) if coord not in ds.dims]
    for col in cols:
        df_bands[col] = ds[col].to_numpy()
    # Get bands from original data (bandpow column)
    f_bands = ds["f_bands"].to_numpy()

    # Plot
    fig_list = []
    for band in f_bands:
        fig_list.append(
            px.box(
                df_bands, x=group, y=band, points="all", title=f"{band} band comparison"
            )
        )
    return fig_list


def get_median_diff(
    data_array: xr.DataArray, group_coord: str, weights_coord: str, dim: str, disp=False
):
    # group_coord: coordinate with groups to compare
    # dim: dimension where the median wants to be computed (for example: 'frequency' to compute the median for each frequency point)
    groups = np.unique(data_array[group_coord])
    assert len(groups) == 2
    # Convert arrays to weighted array
    array_1 = data_array.where(data_array[group_coord] == groups[0], drop=True)
    weighted_array_1 = DataArrayWeighted(array_1, array_1.coords[weights_coord])
    array_2 = data_array.where(data_array[group_coord] == groups[1], drop=True)
    weighted_array_2 = DataArrayWeighted(array_2, array_2.coords[weights_coord])
    # Compute diff
    diff_array = (
        weighted_array_1.quantile(0.5, dim="n").to_numpy()
        - weighted_array_2.quantile(0.5, dim="n").to_numpy()
    )
    if disp:
        bands = data_array[dim].to_numpy()
        for band, diff in zip(bands, diff_array):
            print(f"Median diff {band}: {diff}")
    return diff_array


def get_permvals(
    data_array: xr.DataArray, group_coord: str, weights_coord: str, dim: str
):
    # First sort based on the coordinate of importance
    sorted_array = data_array.sortby(group_coord)
    # Permute the indexes
    perm_idx = np.random.permutation(data_array.indexes["n"])
    # Permuted array
    perm_array = sorted_array.loc[perm_idx, :].copy()
    # Assign permuted labels
    perm_array = perm_array.assign_coords(
        permlabels=("n", sorted_array[group_coord].copy().to_numpy())
    )
    # Get permuted median diff
    perm_diff = get_median_diff(
        perm_array, "permlabels", weights_coord, dim, disp=False
    )

    del perm_array
    return perm_diff


def permutation_test(
    data_array: xr.DataArray,
    group_coord: str,
    weights_coord: str,
    dim: str,
    n_perm: int,
):
    # group_coord: coordinate with groups to compare
    # dim: dimension where the p-vals wants to be computed (for example: 'frequency' to compute the median for each frequency point)
    # Get original median diff
    print("Median difference for true labels")
    true_cond = get_median_diff(data_array, group_coord, weights_coord, dim, disp=True)
    # Build perm matrix based on number of permutations (n_perm) and the number of labels in dim (number of p-vals to compute)
    permvals = np.zeros((n_perm, len(data_array[dim])))
    for i in range(n_perm):
        permvals[i, :] = get_permvals(data_array, group_coord, weights_coord, dim)

    # method p_c
    p_c = np.sum(np.abs(permvals) > np.abs(true_cond), axis=0) / n_perm
    return p_c, permvals

def eval_significance(p_vals, labels):
    from statsmodels.stats.multitest import fdrcorrection
    print('Significance without correction')
    for p_val, label in zip(p_vals, labels):
        print(f'p-val for {label}: {p_val}')
        if p_val < 0.025: #not so valuable, better do 
            print(f'p-val for {label} is significant')
    print('Significance with correction (FDR)')
    # Better to correct
    corr_significance, corr_p_vals = fdrcorrection(p_vals)
    for corr_p_val, sig, label in zip(corr_p_vals, corr_significance, labels):
        print(f'corrected p-val for {label}: {corr_p_val}')
        if sig: #not so valuable, better do 
            print(f'p-val for {label} is significant using FDR Correction')
