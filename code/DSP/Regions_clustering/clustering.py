import sys
from bids import BIDSLayout
import pandas as pd
import numpy as np
from utils import extract_info, df_per_region, plotAllChannels_time, plotAllChannels_plotly, errorModifiedZScore, save_results_clustering
import plotly.express as px
from sklearn.decomposition import PCA
import os
import matplotlib.pyplot as plt
# Clustering
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


edf_path = str(sys.argv[1])
tsv_path = str(sys.argv[2])
out_path = str(sys.argv[3])

# Load tsv files with electrodes position
# locations_tsv_layout = BIDSLayout(tsv_path, validate = False)
# locations_tsv_files = locations_tsv_layout.get(extension='tsv', suffix='space', return_type='filename')

# Load edf files 
edf_files_layout = BIDSLayout(edf_path, validate=False)
edf_files = edf_files_layout.get(extension='edf', return_type='filename')
edf_images = edf_files_layout.get(extension='edf')

# Extract info per region
signal_regions, psd_regions, f_regions, dict_to_edf = extract_info(tsv_path, edf_files, edf_images)

# Convert dict to dataframe
df_info = pd.DataFrame(dict_to_edf)

# Create output folder if doesn't exist
if not os.path.exists(out_path):
    os.mkdir(out_path)

# print(psd_regions.keys())

for region in psd_regions:
    # print(region)
    # Extract data
    psd = np.vstack(psd_regions[region])
    f = np.vstack(f_regions[region])
    signals = np.vstack(signal_regions[region])
    # Get new dataframe for this specific region
    df_region = df_per_region(df_info, region)
    # Save plots
    out_path_region = os.path.join(out_path,f'{region}')
    # Create dir for region folder
    if not os.path.exists(out_path_region):
        print(out_path_region)
        os.makedirs(out_path_region)
        # Time domain
        time = np.arange(signals.shape[-1])/200 #TODO: change srate
        plotAllChannels_time(time, signals, df_region['ID'].tolist(), out_path=os.path.join(out_path_region, 'time_domain_signals.html'))
        # Freq domain
        plotAllChannels_plotly(f, psd, df_region['ID'].tolist(), out_path=os.path.join(out_path_region, 'freq_domain_signals.html'))

        # Get the median and mean absolute deviation (MAD)
        median = np.median(psd, axis=0)
        MAD_region = np.median(np.abs(np.subtract(psd,median)), axis=0)

        # Calculate errors per freq band
        # Extract data per signal
        errors_list = []
        for idx, psd_tmp in enumerate(psd_regions[region]):
            # print(region)
            # Calculate error per freq band
            freq_bands = [0.5,4,8,13,30,80]
            error_bands = []
            for i in range(len(freq_bands)-1):
                idx0 = np.argmin(np.abs(f_regions[region][idx]-freq_bands[i]))
                idx1 = np.argmin(np.abs(f_regions[region][idx]-freq_bands[i+1]))
                error_bands.append(errorModifiedZScore(median[idx0:idx1], MAD_region[idx0:idx1], psd_tmp[idx0:idx1].squeeze()))
            tmp_dict = {
                'Error Delta': np.round(error_bands[0], decimals=4),
                'Error Theta': np.round(error_bands[1], decimals=4),
                'Error Alpha': np.round(error_bands[2], decimals=4),
                'Error Beta': np.round(error_bands[3], decimals=4),
                'Error Gamma': np.round(error_bands[4], decimals=4)
            }
            errors_list.append(tmp_dict)
        df_errors_region = pd.DataFrame(errors_list)
        # Update df_region
        df_region = pd.concat([df_region, df_errors_region], axis=1)

        # Plot PCA output with 3 components
        errors = df_region[['Error Delta', 'Error Theta', 'Error Alpha', 'Error Beta', 'Error Gamma']].to_numpy()
        PCA_vis = PCA(n_components = 3)
        psd_vis = PCA_vis.fit_transform(errors)
        features_plot = [f'{k+1} component' for k in range(3)]
        # Create new dataframe to visualize results
        df_plot = pd.DataFrame(data = psd_vis, columns = features_plot)
        # Update df_plot
        df_plot = pd.concat([df_region, df_plot], axis=1)
        # Save fig
        fig2 = px.scatter_3d(df_plot, x='1 component', y='2 component', z='3 component', hover_name='ID')
        fig2.write_html(os.path.join(out_path_region, '3D_scatterplot_PCA.html'), include_mathjax='cdn')
        del fig2

        # k-means
        # Initializing the cluster algorithm
        kmeans_pipe = Pipeline([
            ('scale', StandardScaler()),
            ('kmeans', KMeans(n_clusters=4))
        ])

        kmeans_pipe.fit(errors)
        clustering_type = 'K_means'
        df_plot[clustering_type] = kmeans_pipe.predict(errors)
        # Plot and save results
        save_results_clustering(clustering_type, out_path_region, df_plot, psd, f)
        

        ## GMM
        clustering_type = 'GMM'
        gmm_pipe = Pipeline([
            ('scale', StandardScaler()),
            ('gm', GaussianMixture(n_components=4, random_state=0))
        ])
        df_plot[clustering_type] = gmm_pipe.fit_predict(errors)
        # Plot and save results
        save_results_clustering(clustering_type, out_path_region, df_plot, psd, f)

