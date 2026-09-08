import pandas as pd
import numpy as np

def filter_missingness(df_values, min_samples, min_int):
    # Extract sample names and intensities
    # df_ind = df['Labels']
    # df_values = df.iloc[:, 1:]

    # Loop convert min_samples if float
    if isinstance(min_samples, float):
        min_samples = int(df_values.shape[0] * min_samples)
    print('min_samples:', min_samples)

    # Flag columns that are BELOW the threshold
    ind_cols_keep = np.full(shape=df_values.shape[1], fill_value=np.nan)
    cols = list(df_values.columns)
    for i, col in enumerate(cols):
        df_sel = np.array(df_values[col])
        arr_above = df_sel[df_sel > min_int]
        if len(arr_above) >= min_samples:
            np.put(a=ind_cols_keep, ind=i, v=i)
    ind_cols_keep = ind_cols_keep[ind_cols_keep == ind_cols_keep]

    ind_cols_keep = [int(el) for el in ind_cols_keep]
    cols_keep = [col for i, col in enumerate(cols) if i in ind_cols_keep]
    df_out = df_values.loc[:, cols_keep]

    return df_out

fileloc = 'C:\\Users\\dirkw\\PycharmProjects\\pythonProject\\PhD\\UPCI\\OutputData\\'
filename_UPCI = '20260707 UPCI GCTAp output (HepG2+HEK, with UPCI).csv'
filename_noUPCI = '20260707 UPCI GCTAp output (HepG2+HEK, no UPCI).csv'

# df = pd.read_csv(fileloc + filename_UPCI)
# print('Before filtering, with UPCI', df.shape)
# df_filt = filter_missingness(df, min_samples=0.25, min_int=5*10**4)
# print('After filtering, with UPCI', df_filt.shape)
#
# df = pd.read_csv(fileloc + filename_noUPCI)
# print('Before filtering, no UPCI', df.shape)
# df_filt = filter_missingness(df, min_samples=0.25, min_int=5*10**4)
# print('After filtering, no UPCI', df_filt.shape)

