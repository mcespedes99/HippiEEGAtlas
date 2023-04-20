import pyedflib
import numpy as np
import os
from multiprocessing.pool import Pool

def extract_length(edf_file):
    edf_in = pyedflib.EdfReader(edf_file)
    # Get number of samples
    N = edf_in.getNSamples()[0]
    return N

def main():
    edf_files = snakemake.input.edf
    csv_out = snakemake.output.out_csv
    processes = int(snakemake.threads)

    # Read edf files
    with Pool(processes=processes) as pool:
        N_total = pool.map(extract_length, 
                           edf_files)
        
    # Write csv file with length values
    np.savetxt(csv_out, np.array(N_total), delimiter="\t")

if __name__=="__main__":
    main()