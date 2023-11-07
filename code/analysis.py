import numpy as np
import xarray as xr
from xarray.core.weighted import DataArrayWeighted
import plotly.express as px
import pandas as pd


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
