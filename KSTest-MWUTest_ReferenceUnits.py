import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from UPCIFunctions.CalcMassDefect import calc_mass_defect
from UPCIFunctions.CalcDataframeSizes import calc_dataframe_sizes
from scipy.stats import ks_2samp, mannwhitneyu
from numpy.random import seed
from matplotlib.gridspec import GridSpec

# Adjustable parameters
fileloc_DB = 'C:\\Users\\dirkw\\PycharmProjects\\pythonProject\\PhD\\UPCI\\DatabaseFilesProcessed\\'
filename_hits = '20260505 HMDB+LM entries incl isotopes (H+).csv'
filename_noise = '20260505 Random noise (HMDB+LM entries incl isotopes, H+, hits removed).csv'
filename_refunits = 'C:\\Users\\dirkw\\PycharmProjects\\pythonProject\\PhD\\UPCI\\InputData\\ReferenceUnits.xlsx'
fileloc_out = 'C:\\Users\\dirkw\\PycharmProjects\\pythonProject\\PhD\\UPCI\\OutputData\\'
filename_out = '20260727 KS- and MWU-test results for different ref units(HMDB+LM entries incl isotopes, H+, hits removed).xlsx'
seed_val = 0
balance_ratio = 1

seed(seed_val)

# Generate figure
fig = plt.figure(figsize=(10, 20))
gs = GridSpec(ncols=2, nrows=1, figure=fig)
ax_KS = fig.add_subplot(gs[0, 0])
ax_MWU = fig.add_subplot(gs[0, 1])

# Load and select training data, train UPCI model
df_hits = pd.read_csv(fileloc_DB + filename_hits)
df_noise = pd.read_csv(fileloc_DB + filename_noise)
df_refunits = pd.read_excel(filename_refunits)

N_hits, N_noise = calc_dataframe_sizes(N=df_hits.shape[0], N_noise_max=df_noise.shape[0], balance_ratio=balance_ratio)
df_nonhits = df_noise.sample(N_noise, replace=False, ignore_index=True)
df_nonhits.reset_index(inplace=True, drop=True)

mz_hits = np.array(df_hits['m/z'])
mz_noise = np.array(df_nonhits['m/z'])
df_res_cols = ['Ref. unit', 'KS p-value', 'MWU p-value']
df_res = pd.DataFrame(columns=df_res_cols, data=np.full(fill_value=np.nan, shape=(df_refunits.shape[0], len(df_res_cols))))

for i in range(df_refunits.shape[0]):
    df_row = df_refunits.iloc[i, :]
    ref_name_i, ref_mass_i = df_row['Name'], df_row['Mass']

    # Calculate mass defects
    KMD_arr_hits = calc_mass_defect(mz_hits, ref_mass_i)
    KMD_arr_noise = calc_mass_defect(mz_noise, ref_mass_i)

    # Perform KS-test and MWU-test
    ks_statistic, KS_p_value = ks_2samp(data1=KMD_arr_hits, data2=KMD_arr_noise, alternative='two-sided')
    MWU_statistic, MWU_p_value = mannwhitneyu(x=KMD_arr_hits, y=KMD_arr_noise, alternative='two-sided')

    # Save results
    df_res.loc[i, 'Ref. unit'] = ref_name_i
    df_res.loc[i, 'KS p-value'] = KS_p_value
    df_res.loc[i, 'MWU p-value'] = MWU_p_value

# Store final output
df_res.to_excel(fileloc_out+filename_out, index=False)