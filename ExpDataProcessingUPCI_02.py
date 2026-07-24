import pandas as pd
import numpy as np
from UPCIFunctions.FilterMissingness import filter_missingness

# Adjustable parameters
fileloc = 'C:\\Users\\dirkw\\PycharmProjects\\pythonProject\\PhD\\UPCI\\OutputData\\'
filename_UPCI = '20260710 UPCI GCTAp output (EV batch 2, no UPCI).csv'
filename_out = '20260713 UPCI GCTAp output (EV batch 2, no UPCI, preprocessed).csv'
CoV_threshold = 0.30
I_threshold = 10 ** 3
min_samples = 0.25 * 0.2
min_int = 50 * 10 ** 3

# Load data
df_UPCI = pd.read_csv(fileloc+filename_UPCI)

# Add types
df_labels = list(df_UPCI['Labels'])
labels = []
for i in range(len(df_labels)):
    if '5637' in df_labels[i]:
        labels += ['5637']
    elif 'B_' in df_labels[i]:
        labels += ['Blank']
    elif 'HT1376' in df_labels[i]:
        labels += ['HT1376']
    elif 'RT112' in df_labels[i]:
        labels += ['RT112']
    elif 'RT4' in df_labels[i]:
        labels += ['RT4']
    elif 'UMUC' in df_labels[i]:
        labels += ['UMUC3']
df_UPCI['Type'] = labels

# Flag columns that are BELOW the threshold
df_values = df_UPCI.iloc[:, 1:-1].copy()
df_values = filter_missingness(df_values, min_int=min_int, min_samples=min_samples)
df_values_blank = df_values[df_UPCI['Type'] == 'Blank']
df_values_nonblank = df_values[df_UPCI['Type'] != 'Blank']

# ind_cols_keep = np.full(shape=df_values.shape[1], fill_value=np.nan)
# cols = list(df_values.columns)
# for i, col in enumerate(cols):
#     # Load data, generate bools for filtering
#     df_sel = np.array(df_values[col])
#     bool_CoV = False
#     bool_nancnt = False
#     bool_int = False
#
#     # Remove columns with NaNs
#     cnt_nan = np.count_nonzero(np.isnan(df_sel))
#     if cnt_nan == 0:
#         bool_nancnt = True
#         # np.put(a=ind_cols_keep, ind=i, v=i)
#
#     # Keep features with CoV above specified threshold
#     CoV = np.nanstd(df_sel) / np.nanmean(df_sel)
#     if CoV >= CoV_threshold:
#         bool_CoV = True
#         # np.put(a=ind_cols_keep, ind=i, v=i)
#
#     # Keep features with all intensities above specified threshold
#     df_sel_above = df_sel[df_sel > I_threshold]
#     if len(df_sel_above) == len(df_sel):
#         bool_int = True
#
#     if bool_int and bool_CoV and bool_nancnt:
#         np.put(a=ind_cols_keep, ind=i, v=i)

# Only keep columns in list
# ind_cols_keep = ind_cols_keep[ind_cols_keep == ind_cols_keep]
# ind_cols_keep = [int(el) for el in ind_cols_keep]
# cols_keep = [col for i, col in enumerate(cols) if i in ind_cols_keep]
# df_prep = df_values.loc[:, cols_keep]

# Perform blank subtraction
df_prep_blank = df_values[df_UPCI['Type'] == 'Blank']
df_prep_nonblank = df_values[df_UPCI['Type'] != 'Blank']
df_prep_nonblank_labels = list(df_UPCI['Type'][df_UPCI['Type'] != 'Blank'])
df_prep_nonblank = df_prep_nonblank - np.median(df_prep_blank, axis=0)

# Replace negative values with NaN
df_prep_nonblank[df_prep_nonblank < 0] = np.nan

# Impute random values for missing values
random_values = np.random.uniform(low=1, high=100, size=df_prep_nonblank.shape)
df_prep_nonblank = df_prep_nonblank.fillna(pd.DataFrame(data=random_values, index=df_prep_nonblank.index, columns=df_prep_nonblank.columns))

# Perform autoscaling
arr_std = np.std(df_prep_nonblank, axis=0)
arr_mean = np.mean(df_prep_nonblank, axis=0)
df_prep_nonblank = (df_prep_nonblank - arr_mean) / arr_std

# Store output
df_prep_nonblank['Type'] = df_prep_nonblank_labels
df_prep_nonblank.to_csv(fileloc+filename_out, index=False)