import pandas as pd
import numpy as np
from fisher_py import RawFile
from UPCIFunctions.CalcMassDefect import calc_mass_defect
from UPCIFunctions.UPCI_Blacklist import UPCI_blacklist
from UPCIFunctions.CalcDataframeSizes import calc_dataframe_sizes
from PhD.GeneralFunctions.ExtractScanPropertiesRawFile import extract_scan_properties
from PhD.GeneralFunctions.GCTAn_Python import GCTAn_Python
from PhD.GeneralFunctions.GCTAp_Python import GCTAp_Python
from PhD.GeneralFunctions.GCTA_Centroid_SavGol_Python import filter_centroid_single_scan
from datetime import datetime
from numpy.random import seed
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

pd.set_option('display.max_rows', None)

def apply_UPCI_model_v2(model, df_data_in, P_thresh, ref_masses, mz_col='m/z'):
    df_data_UPCI = pd.DataFrame()
    df_data_UPCI['m/z'] = df_data_in[mz_col].copy()
    for l, ref_mass in enumerate(ref_masses):
        df_data_UPCI['KMD ' + str(l + 1)] = calc_mass_defect(np.array(df_data_UPCI['m/z']), ref_mass=ref_mass)
    df_probs = pd.DataFrame(data=model.predict_proba(df_data_UPCI), columns=model.classes_)
    df_probs_cols = [str(el) for el in list(df_probs.columns)]
    df_probs.columns = df_probs_cols
    df_data_UPCI['Prob. hit'] = np.array(df_probs['1'])
    df_data_out = df_data_in[df_data_UPCI['Prob. hit'] >= P_thresh]
    df_data_removed = df_data_in[df_data_UPCI['Prob. hit'] < P_thresh]
    return df_data_out, df_data_removed

# Adjustable variables
MSr_filenames = open('PCAPlot_UPCIEffect_ExpData_Filenames_HEK-HepG2.txt', 'r').readlines()
fileloc_DB = 'C:\\Users\\dirkw\\PycharmProjects\\pythonProject\\PhD\\UPCI\\DatabaseFilesProcessed\\'
filename_hits = '20260505 HMDB+LM entries incl isotopes (H+).csv'
filename_nonhits = '20260505 Random noise (HMDB+LM entries incl isotopes, H+, hits removed).csv'
filename_blacklist = 'C:\\Users\\dirkw\\PycharmProjects\\pythonProject\\PhD\\UPCI\\InputData\\20260317 MS contaminants (processed).xlsx'
MS_fileloc = 'C:\\Users\\dirkw\\PycharmProjects\\pythonProject\\PhD\\UPCI\\MSDataRaw\\'

fileloc_out = 'C:\\Users\\dirkw\\PycharmProjects\\pythonProject\\PhD\\UPCI\\OutputData\\'
filename_noUPCI_GCTAp = '20260710 UPCI GCTAp output (HEK+HepG2, no UPCI).csv'
filename_UPCI_GCTAp = '20260710 UPCI GCTAp output (HEK+HepG2, with UPCI).csv'

seed_val = 0
polarity = '+'

# GCTA parameters
GCTAn_K = 110
GCTAn_threshold = 3
GCTAn_eps = 3
GCTAp_eps = 5

# Experimental variables
seed(seed_val)
ref_masses = [14.01565]
balance_ratio = 1
max_depth = 20
UPCI_pos = 2
P_thresh = 0.35
MSr_filenames = [el.strip() for el in MSr_filenames]

# Load and select training data, train UPCI model
df_hits = pd.read_csv(fileloc_DB + filename_hits)
df_nonhits = pd.read_csv(fileloc_DB + filename_nonhits)

N_hits, N_noise = calc_dataframe_sizes(N=df_hits.shape[0], N_noise_max=df_nonhits.shape[0], balance_ratio=balance_ratio)
df_nonhits = df_nonhits.sample(N_noise, replace=False, ignore_index=True)
df_nonhits.reset_index(inplace=True, drop=True)

df_train_hits = pd.DataFrame()
df_train_hits['m/z'] = df_hits['m/z'].copy()
for j, ref_mass in enumerate(ref_masses):
    df_train_hits['KMD ' + str(j + 1)] = calc_mass_defect(np.array(df_train_hits['m/z']), ref_mass=ref_mass)

df_train_nonhits = pd.DataFrame()
df_train_nonhits['m/z'] = df_nonhits['m/z'].copy()
for j, ref_mass in enumerate(ref_masses):
    df_train_nonhits['KMD ' + str(j + 1)] = calc_mass_defect(np.array(df_train_nonhits['m/z']), ref_mass=ref_mass)

df_train_hits['Label'] = np.full(shape=df_hits.shape[0], fill_value=1)
df_train_nonhits['Label'] = np.full(shape=df_nonhits.shape[0], fill_value=0)
df_train = pd.concat([df_train_hits, df_train_nonhits], axis=0)
print(datetime.now().strftime("%H:%M:%S") + ' | Finished loading and generating training data set')

# Train UPCI model
X_train, y_train = df_train.iloc[:, :-1].values, df_train.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1, random_state=0)
model = RandomForestClassifier(max_depth=max_depth, random_state=seed_val)
model.fit(X_train, y_train)
print(datetime.now().strftime("%H:%M:%S") + ' | Finished training UPCI model')

# Load data for blacklisting
df_blacklist = pd.read_excel(filename_blacklist)
mz_blacklist = np.array(df_blacklist['m/z'])

df_GCTAn_noUPCI = pd.DataFrame()
df_GCTAn_UPCI = pd.DataFrame()

for i, MS_filename in enumerate(MSr_filenames):
    print(datetime.now().strftime("%H:%M:%S") + ' | Processing MS file', i+1, 'of', len(MSr_filenames))
    # Load raw file
    raw_file = RawFile(MS_fileloc + MS_filename)

    # Extract scan properties
    df_scanprops = extract_scan_properties(raw_file)
    df_scanprops = df_scanprops[df_scanprops['Polarity'] == polarity]
    df_scanprops = df_scanprops[df_scanprops['MS2 central m/z'] != df_scanprops['MS2 central m/z']]  # Remove MS2 scans
    scan_nrs = np.array(df_scanprops['Scan nr.'])
    print(datetime.now().strftime("%H:%M:%S") + ' | Finished extracting scan properties')

    # Load MS data to dataframe, apply filtering and centroiding
    df_centroid_cols = ['Scan nr.', 'm/z', 'I']
    df_centroid = pd.DataFrame(columns=df_centroid_cols)
    for scan_nr in scan_nrs:
        df_centroid_new = pd.DataFrame(columns=df_centroid_cols)
        mz, I = raw_file.get_scan_from_scan_number(int(scan_nr))[0], raw_file.get_scan_from_scan_number(int(scan_nr))[1]
        mz_centroid, I_centroid = filter_centroid_single_scan(mz, I)
        df_centroid_new['m/z'] = mz_centroid
        df_centroid_new['I'] = I_centroid
        df_centroid_new['Scan nr.'] = np.full(fill_value=scan_nr, shape=df_centroid_new.shape[0])
        df_centroid = pd.concat([df_centroid, df_centroid_new], axis=0)
    print(datetime.now().strftime("%H:%M:%S") + ' | Finished loading, filtering and centroiding MS data')

    # Apply GCTAn (clustering)
    mz_in, I_in = np.array(df_centroid['m/z']), np.array(df_centroid['I'])
    df_GCTAn_file, _ = GCTAn_Python(mz_in, I_in, K=GCTAn_K, eps=GCTAn_eps, threshold=GCTAn_threshold)
    print(datetime.now().strftime("%H:%M:%S") + ' | Finished GCTAn clustering')

    # Apply blacklist filtering
    df_GCTAn_file, _ = UPCI_blacklist(df_GCTAn_file, mz_colname='m/z', mz_blacklist=mz_blacklist)
    df_GCTAn_file.reset_index(inplace=True)

    # Store GCTAn output for GCTAp analysis
    df_GCTAn_file['Filename'] = df_GCTAn_file.shape[0] * [MS_filename]
    df_GCTAn_noUPCI = pd.concat([df_GCTAn_noUPCI, df_GCTAn_file], axis=0)

    if UPCI_pos == 1:
        print(datetime.now().strftime("%H:%M:%S") + ' | Applying UPCI')
        # Apply UPCI
        df_centroid, _ = apply_UPCI_model_v2(model=model, df_data_in=df_centroid, P_thresh=P_thresh, ref_masses=ref_masses)

        # Apply GCTAn
        mz_in, I_in = np.array(df_centroid['m/z']), np.array(df_centroid['I'])
        df_GCTAn_file, _ = GCTAn_Python(mz_in, I_in, K=GCTAn_K, eps=GCTAn_eps, threshold=GCTAn_threshold)
        print(datetime.now().strftime("%H:%M:%S") + ' | Finished GCTAn clustering')

        # Apply blacklist filtering
        df_GCTAn_file, _ = UPCI_blacklist(df_GCTAn_file, mz_colname='m/z', mz_blacklist=mz_blacklist)
        df_GCTAn_file.reset_index(inplace=True)

        # Store GCTAn output for GCTAp analysis
        df_GCTAn_file['Filename'] = df_GCTAn_file.shape[0] * [MS_filename]
        df_GCTAn_UPCI = pd.concat([df_GCTAn_UPCI, df_GCTAn_file], axis=0)

    elif UPCI_pos == 2:
        print(datetime.now().strftime("%H:%M:%S") + ' | Applying UPCI')
        # Apply GCTAn (clustering)
        mz_in, I_in = np.array(df_centroid['m/z']), np.array(df_centroid['I'])
        df_GCTAn_file, _ = GCTAn_Python(mz_in, I_in, K=GCTAn_K, eps=GCTAn_eps, threshold=GCTAn_threshold)
        print(datetime.now().strftime("%H:%M:%S") + ' | Finished GCTAn clustering')

        # Apply UPCI
        df_GCTAn_file, _ = apply_UPCI_model_v2(model=model, df_data_in=df_GCTAn_file, P_thresh=P_thresh, ref_masses=ref_masses)

        # Apply blacklist filtering
        df_GCTAn_file, _ = UPCI_blacklist(df_GCTAn_file, mz_colname='m/z', mz_blacklist=mz_blacklist)
        df_GCTAn_file.reset_index(inplace=True)

        # Store GCTAn output for GCTAp analysis
        df_GCTAn_file['Filename'] = df_GCTAn_file.shape[0] * [MS_filename]
        df_GCTAn_UPCI = pd.concat([df_GCTAn_UPCI, df_GCTAn_file], axis=0)

# Apply GCTAp (no UPCI)
mz_in, I_in, labels_in = np.array(df_GCTAn_noUPCI['m/z']), np.array(df_GCTAn_noUPCI['Intensity']), np.array(df_GCTAn_noUPCI['Filename'])
df_GCTAp_noUPCI = GCTAp_Python(mz_in, I_in, eps=GCTAp_eps, labels=labels_in)

# Apply GCTAp (with UPCI)
if UPCI_pos == 1 or UPCI_pos == 2:
    mz_in, I_in, labels_in = np.array(df_GCTAn_UPCI['m/z']), np.array(df_GCTAn_UPCI['Intensity']), np.array(df_GCTAn_UPCI['Filename'])
    df_GCTAp_UPCI = GCTAp_Python(mz_in, I_in, eps=GCTAp_eps, labels=labels_in)
elif UPCI_pos == 3:
    print(datetime.now().strftime("%H:%M:%S") + ' | Applying UPCI')
    df_GCTAp_UPCI, _ = apply_UPCI_model_v2(model=model, df_data_in=df_GCTAp_noUPCI, P_thresh=P_thresh, ref_masses=ref_masses, mz_col='m/z aligned')

# Process GCTAp output (with and without UPCI), save output
print(datetime.now().strftime("%H:%M:%S") + ' | Processing GCTAp output (no UPCI)')
df_matrix_noUPCI = df_GCTAp_noUPCI.pivot_table(index="Labels", columns="m/z aligned", values="Intensity", aggfunc="max")
df_matrix_noUPCI.to_csv(fileloc_out+filename_noUPCI_GCTAp)
print(datetime.now().strftime("%H:%M:%S") + ' | Processing GCTAp output (with UPCI)')
df_matrix_UPCI = df_GCTAp_UPCI.pivot_table(index="Labels", columns="m/z aligned", values="Intensity", aggfunc="max")
df_matrix_UPCI.to_csv(fileloc_out+filename_UPCI_GCTAp)