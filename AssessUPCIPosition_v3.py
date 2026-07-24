import pandas as pd
import numpy as np
from fisher_py import RawFile
from UPCIFunctions.CalcMassDefect import calc_mass_defect
from UPCIFunctions.UPCI_Blacklist import UPCI_blacklist
from UPCIFunctions.CalcDataframeSizes import calc_dataframe_sizes
from PhD.GeneralFunctions.ExtractScanPropertiesRawFile import extract_scan_properties
from PhD.GeneralFunctions.GCTAn_Python import GCTAn_Python
from PhD.GeneralFunctions.GCTAp_Python import GCTAp_Python
from UPCIFunctions.SearchDatabase import search_database
from PhD.GeneralFunctions.GCTA_Centroid_SavGol_Python import filter_centroid_single_scan
from datetime import datetime
from numpy.random import seed
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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
MSr_filenames = open('AssessUPCIPosition_MSrFilenames_final_HEK-HepG2.txt', 'r').readlines()
metrics_descr = open('AssessUPCIPosition_Metrics_v3.txt', 'r').readlines()
seed_val = 0
fileloc_res_xlsx = 'C:\\Users\\dirkw\\PycharmProjects\\pythonProject\\PhD\\UPCI\\UPCIPositionAssessmentResults\\XLSX (v3)\\'
fileloc_res_csv = 'C:\\Users\\dirkw\\PycharmProjects\\pythonProject\\PhD\\UPCI\\UPCIPositionAssessmentResults\\CSV (v3)\\'
writer_res_filename = '20260714 UPCI positioning assessment results (v3, benchmark, HEK+HepG2, HMDB+LM hits and noise)'
save_output = True
runs_finished = 0

# GCTA parameters
GCTAn_K = 110
GCTAn_threshold = 3
GCTAn_eps = 3
GCTAp_eps = 5

# Experimental variables
seed(seed_val)
df_pars = pd.read_excel('C:\\Users\\dirkw\\PycharmProjects\\pythonProject\\PhD\\UPCI\\InputData\\InputPars_AssessmentUPCIPosition_v3.xlsx')
polarity = '+'

MSr_filenames = [el.strip() for el in MSr_filenames]
metrics_descr = [el.strip() for el in metrics_descr]
if runs_finished == 0:
    df_res_main = pd.DataFrame()
else:
    df_res_main = pd.read_csv(fileloc_res_csv+writer_res_filename + ' (metrics).csv')
    print(df_res_main)

# Extract scan properties, perform centroiding and filtering
t_start_analyses = datetime.now()

for i in range(df_pars.shape[0]):
    if i > 0:
        ETF_analyses = (datetime.now() - t_start_analyses) / i * df_pars.shape[0]
        print(datetime.now().strftime("%H:%M:%S") + ' | Processing analysis', i + 1, 'of', df_pars.shape[0], '| ETF: ' + (t_start_analyses + ETF_analyses).strftime("%d %B, %H:%M:%S"))
    else:
        print(datetime.now().strftime("%H:%M:%S") + ' | Processing analysis', i + 1, 'of', df_pars.shape[0])

    run_ID = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_ID_list = [run_ID] * len(MSr_filenames)
    t_start = datetime.now()

    # Load parameters from input dataframe
    df_pars_row = df_pars.iloc[i, :]
    fileloc_input = df_pars_row['fileloc_input']
    filename_hits = df_pars_row['filename_hits']
    filename_nonhits = df_pars_row['filename_nonhits']
    fileloc_DB = df_pars_row['fileloc_DB']
    filename_DB = df_pars_row['filename_DB']
    MS_fileloc = df_pars_row['MS_fileloc']
    UPCI_pos = df_pars_row['UPCI_position']
    max_depth = df_pars_row['max_depth']
    P_thresh = df_pars_row['P_thresh']
    ref_masses = df_pars_row['KMD_ref_masses']
    balance_ratio = df_pars_row['balance_ratio']
    filename_blacklist = df_pars_row['filename_blacklist']

    # Convert reference masses to list of floats
    if not isinstance(ref_masses, float):
        ref_masses = [float(el) for el in ref_masses.split(';')]
    else:
        ref_masses = [ref_masses]

    if save_output:
        writer_res = pd.ExcelWriter(fileloc_res_xlsx + writer_res_filename + ' - Run ' + str(i + runs_finished + 1) + ' (output, ' + run_ID + ').xlsx', engine='openpyxl')
        writer_I_kept = pd.ExcelWriter(fileloc_res_xlsx + writer_res_filename + ' - Run ' + str(i + runs_finished + 1) + ' (I kept, '+run_ID+').xlsx', engine='openpyxl')
        writer_I_removed = pd.ExcelWriter(fileloc_res_xlsx + writer_res_filename + ' - Run ' + str(i + runs_finished + 1) + ' (I removed, '+run_ID+').xlsx', engine='openpyxl')

    # Generate dataframe for output
    df_pars_cols = list(df_pars.columns)
    df_res = pd.DataFrame(data=np.full(shape=(len(MSr_filenames), df_pars.shape[1]), fill_value=np.nan), dtype='object', columns=df_pars_cols)
    for row_nr in range(len(MSr_filenames)):
        for col_nr in range(df_pars.shape[1]):
            df_res.iat[row_nr, col_nr] = df_pars_row.iloc[col_nr]

    # Load and select training data, train UPCI model
    df_hits = pd.read_csv(fileloc_input + filename_hits)
    df_nonhits = pd.read_csv(fileloc_input + filename_nonhits)

    N_hits, N_noise = calc_dataframe_sizes(N=df_hits.shape[0], N_noise_max=df_nonhits.shape[0], balance_ratio=balance_ratio)
    df_nonhits = df_nonhits.sample(N_noise, replace=False, ignore_index=True)
    df_nonhits.reset_index(inplace=True, drop=True)

    df_train_hits = pd.DataFrame()
    df_train_hits['m/z'] = df_hits['m/z'].copy()
    for j, ref_mass in enumerate(ref_masses):
        df_train_hits['KMD '+str(j+1)] = calc_mass_defect(np.array(df_train_hits['m/z']), ref_mass=ref_mass)

    df_train_nonhits = pd.DataFrame()
    df_train_nonhits['m/z'] = df_nonhits['m/z'].copy()
    for j, ref_mass in enumerate(ref_masses):
        df_train_nonhits['KMD '+str(j+1)] = calc_mass_defect(np.array(df_train_nonhits['m/z']), ref_mass=ref_mass)

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

    # y_pred = model.predict(X_test)
    # print(accuracy_score(y_test, y_pred))

    # Load database file and blacklist file
    db_file = pd.read_csv(fileloc_DB+filename_DB)
    db_file.drop_duplicates(subset='m/z', inplace=True)

    df_blacklist = pd.read_excel(fileloc_input+filename_blacklist)
    mz_blacklist = np.array(df_blacklist['m/z'])
    print(datetime.now().strftime("%H:%M:%S") + ' | Finished loading files for DB search and blacklisting')

    df_GCTAn = pd.DataFrame()
    df_sizes = pd.DataFrame(columns=metrics_descr, data=np.zeros(shape=(len(MSr_filenames), len(metrics_descr))))

    for j, MS_file in enumerate(MSr_filenames):
        print(datetime.now().strftime("%H:%M:%S") + ' | PROCESSING MS DATA FILE', MS_file)
        sizes_arr = np.zeros(shape=len(metrics_descr))

        # Load raw file
        raw_file = RawFile(MS_fileloc+MS_file)

        # Extract scan properties
        df_scanprops = extract_scan_properties(raw_file)
        df_scanprops = df_scanprops[df_scanprops['Polarity'] == polarity]
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
        np.put(a=sizes_arr, ind=0, v=df_centroid.shape[0])
        print(datetime.now().strftime("%H:%M:%S") + ' | Finished loading, filtering and centroiding MS data')

        if UPCI_pos == 1:
            df_centroid, df_centroid_removed = apply_UPCI_model_v2(model, df_centroid, P_thresh, ref_masses)
            np.put(a=sizes_arr, ind=1, v=df_centroid.shape[0])
            if save_output:
                df_centroid.to_excel(writer_I_kept, sheet_name=MS_file, index=False)
                df_centroid_removed.to_excel(writer_I_removed, sheet_name=MS_file, index=False)
            print(datetime.now().strftime("%H:%M:%S") + ' | Finished applying UPCI model')

        # Apply GCTAn (clustering)
        mz_in, I_in = np.array(df_centroid['m/z']), np.array(df_centroid['I'])
        df_GCTAn_file, _ = GCTAn_Python(mz_in, I_in, K=GCTAn_K, eps=GCTAn_eps, threshold=GCTAn_threshold)
        if save_output:
            df_GCTAn_file.to_excel(writer_res, sheet_name='GCTAn - Run '+str(i+1)+'-'+str(j+1), index=False)
        np.put(a=sizes_arr, ind=2, v=df_GCTAn_file.shape[0])
        print(datetime.now().strftime("%H:%M:%S") + ' | Finished GCTAn clustering')

        if UPCI_pos == 2:
            df_GCTAn_file, df_GCTAn_file_removed = apply_UPCI_model_v2(model, df_GCTAn_file, P_thresh, ref_masses)
            np.put(a=sizes_arr, ind=1, v=df_GCTAn_file.shape[0])
            if save_output:
                df_GCTAn_file.to_excel(writer_I_kept, sheet_name=MS_file, index=False)
                df_GCTAn_file_removed.to_excel(writer_I_removed, sheet_name=MS_file, index=False)
            print(datetime.now().strftime("%H:%M:%S") + ' | Finished applying UPCI model')

        # Store GCTAn output
        df_GCTAn_file['Filename'] = df_GCTAn_file.shape[0] * [MS_file]
        df_GCTAn = pd.concat([df_GCTAn, df_GCTAn_file], axis=0)

        # Apply blacklist filtering
        df_data, _ = UPCI_blacklist(df_GCTAn_file, mz_colname='m/z', mz_blacklist=mz_blacklist)
        df_data.reset_index(inplace=True)
        np.put(a=sizes_arr, ind=3, v=df_data.shape[0])

        # Perform database search on GCTAn output
        df_GCTAn_file.reset_index(inplace=True)
        df_DBsearch_hits, df_DBsearch_nonhits = search_database(db_file, df_GCTAn_file, data_col_mz='m/z', DB_col_mz='m/z', col_comp='Isotopic composition', col_ID='Isotopic composition', info=False)
        if save_output:
            df_DBsearch_hits.to_excel(writer_res, sheet_name='GCTAn DB search - Run '+str(i+1)+'-'+str(j+1), index=False)
        print(datetime.now().strftime("%H:%M:%S") + ' | Finished GCTAn database search')
        np.put(a=sizes_arr, ind=4, v=df_DBsearch_hits.shape[0])

        # Store array sizes in dataframe
        for k, size in enumerate(sizes_arr):
            df_sizes.iat[j, k] = size

    # Apply GCTAp
    mz_in, I_in, labels_in = np.array(df_GCTAn['m/z']), np.array(df_GCTAn['Intensity']), np.array(df_GCTAn['Filename'])
    df_GCTAp = GCTAp_Python(mz_in, I_in, eps=GCTAp_eps, labels=labels_in)

    if UPCI_pos == 3:
        df_GCTAp, df_GCTAp_removed = apply_UPCI_model_v2(model, df_GCTAp, P_thresh, ref_masses, mz_col='m/z aligned')
        df_sizes['Nr. of data points after UPCI'] = np.full(fill_value=df_GCTAp.shape[0], shape=len(MSr_filenames))
        # if save_output:
        #     df_GCTAp.to_excel(writer_I_kept, sheet_name='Kept GCTAp output', index=False)
        #     df_GCTAp_removed.to_excel(writer_I_removed, sheet_name='Removed GCTAp output', index=False)
        print(datetime.now().strftime("%H:%M:%S") + ' | Finished applying UPCI model')

    if save_output:
        # df_GCTAp.to_excel(writer_res, sheet_name='GCTAp - Run ' + str(i+1), index=False)
        df_GCTAp.to_csv(fileloc_res_xlsx+writer_res_filename+' (GCTAp output - Run ' + str(i+1+runs_finished)+', '+run_ID+').csv', index=False)
        if UPCI_pos == 3:
            df_GCTAp_removed.to_csv(fileloc_res_xlsx+writer_res_filename+' (GCTAp output removed - Run ' + str(i+1+runs_finished)+', '+run_ID+').csv', index=False)
    print(datetime.now().strftime("%H:%M:%S") + ' | Finished running GCTAp')
    df_sizes['Nr. of GCTAp peaks'] = np.full(fill_value=df_GCTAp.shape[0], shape=df_sizes.shape[0])

    # Perform database search on GCTAp output
    df_GCTAp.drop_duplicates(subset='m/z aligned', inplace=True)
    df_GCTAp.reset_index(inplace=True)
    df_DBsearch_hits, df_DBsearch_nonhits = search_database(db_file, df_GCTAp, data_col_mz='m/z aligned', DB_col_mz='m/z', col_comp='Isotopic composition', col_ID='Isotopic composition', info=False)
    if save_output:
        df_DBsearch_hits.to_excel(writer_res, sheet_name='GCTAp DB search - Run ' + str(i + 1), index=False)
    print(datetime.now().strftime("%H:%M:%S") + ' | Finished GCTAp database search')
    df_sizes['Nr. of database hits for GCTAp output'] = np.full(fill_value=df_DBsearch_hits.shape[0], shape=df_sizes.shape[0])

    print(datetime.now().strftime("%H:%M:%S") + ' | Storing output')
    df_res['Run IDs'] = run_ID_list
    df_res = pd.concat([df_res, df_sizes], axis=1)
    df_res_main = pd.concat([df_res_main, df_res], axis=0)
    df_res_main.to_csv(fileloc_res_csv+writer_res_filename + ' (metrics).csv', index=False)
    print(df_res_main)

    # Write results to writer
    if save_output:
        df_res.to_excel(writer_res, sheet_name='Run ' + str(i + runs_finished + 1) + ' - Metrics', index=False)
        writer_res.close()
        if not UPCI_pos == 3:
            writer_I_kept.close()
            writer_I_removed.close()