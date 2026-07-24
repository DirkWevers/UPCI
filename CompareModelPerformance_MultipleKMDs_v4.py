import pandas as pd
import numpy as np
from itertools import combinations
from random import seed, sample
from UPCIFunctions.CalcMassDefect import calc_mass_defect
from UPCIFunctions.CalcDataframeSizes import calc_dataframe_sizes
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, permutation_test_score
from datetime import datetime

seed(0)

# Adjustable parameters
fileloc_refs = 'C:\\Users\\dirkw\\PycharmProjects\\pythonProject\\PhD\\UPCI\\InputData\\'
filename_refs = 'ReferenceUnits.xlsx'
filename_out = '202605 RF performance for multiple KMDs, v4, max depth '
fileloc_out = 'C:\\Users\\dirkw\\PycharmProjects\\pythonProject\\PhD\\UPCI\\OutputData\\'
N_perm = 10
permutation_test = False
info = True

# Load dataframe with input parameters
df_pars = pd.read_excel('C:\\Users\\dirkw\\PycharmProjects\\pythonProject\\PhD\\UPCI\\InputData\\InputPars_RF_MultipleKMDComparison_v4.xlsx')

# Load reference unit data
df_refs = pd.read_excel(fileloc_refs+filename_refs)
df_refs.sort_values(inplace=True, by='Mass')
df_refs = df_refs.iloc[:15, :]
refs_names, refs_masses = list(df_refs['Name']), np.array(df_refs['Mass'])

for i in range(df_pars.shape[0]):
    if info:
        print(datetime.now().strftime("%H:%M:%S") + ' | Started analysis', i+1, 'of', df_pars.shape[0])

    # Load parameter values
    df_pars_row = df_pars.iloc[i, :]
    fileloc = df_pars_row['fileloc']
    filename_hits = df_pars_row['filename_hits']
    filename_nonhits = df_pars_row['filename_nonhits']
    N_var = df_pars_row['N_vars']
    N_test = df_pars_row['N_tests']
    balance_ratio = df_pars_row['balance_ratio']
    CV_fold = df_pars_row['CV_fold']
    max_depth = df_pars_row['max_depth']
    if np.isnan(max_depth):
        max_depth = 0
    else:
        max_depth = int(max_depth)

    # Load hits and noise data, select noise data based on balance ratio
    df_hits = pd.read_csv(fileloc + '\\' + filename_hits)
    df_nonhits = pd.read_csv(fileloc + '\\' + filename_nonhits)

    N_hits, N_noise = calc_dataframe_sizes(N=df_hits.shape[0], N_noise_max=df_nonhits.shape[0], balance_ratio=balance_ratio)
    df_nonhits = df_nonhits.sample(N_noise, replace=False, ignore_index=True)
    df_nonhits.reset_index(inplace=True, drop=True)

    # Calculate KMDs
    df_hits_KMD, df_nonhits_KMD = pd.DataFrame(), pd.DataFrame()
    df_hits_KMD['m/z'], df_nonhits_KMD['m/z'] = df_hits['m/z'].copy(), df_nonhits['m/z'].copy()
    for ref_name, ref_mass in zip(refs_names, refs_masses):
        df_hits_KMD[ref_name] = calc_mass_defect(arr=np.array(df_hits['m/z']), ref_mass=ref_mass)
        df_nonhits_KMD[ref_name] = calc_mass_defect(arr=np.array(df_nonhits['m/z']), ref_mass=ref_mass)

    # Generate model
    if max_depth >= 1:
        model = RandomForestClassifier(random_state=0, max_depth=max_depth)
    else:
        model = RandomForestClassifier(random_state=0, max_depth=None)

    # Generate random combinations of reference masses
    t_start_Nvars = datetime.now()
    refs_combs = list(combinations(refs_names, N_var))
    refs_combs = sample(refs_combs, N_test)

    # Create empty variables
    df_res_cols = ['Var '+str(i+1) for i in range(N_var)]
    df_res = pd.DataFrame(columns=df_res_cols)
    arr_vars = np.zeros(shape=(N_test, N_var), dtype='object')
    cv_scores_arr = np.zeros(shape=(N_test, CV_fold))
    perm_pvals = np.full(shape=N_test, fill_value=np.nan)
    perm_score_arr = np.full(shape=N_test, fill_value=np.nan)
    perm_scores_arr = np.full(shape=(N_test, N_perm), fill_value=np.nan)
    times_arr = np.zeros(len(refs_combs))

    t_start_combs = datetime.now()
    for j, refs_comb in enumerate(refs_combs):
        if info and j > 0:
            ETF_combs = (datetime.now() - t_start_combs) / j * len(refs_combs)
            print(datetime.now().strftime("%H:%M:%S") + ' | Processing combination of variables', j + 1, 'of', len(refs_combs), '| ETF: ' + (t_start_combs + ETF_combs).strftime("%d %B, %H:%M:%S"))

        # Save variables to numpy array
        for k, ref in enumerate(refs_comb):
            arr_vars[j, k] = ref

        # Add required KMDs to dataframe
        df_hits_model, df_nonhits_model = pd.DataFrame(), pd.DataFrame()
        df_hits_model['m/z'], df_nonhits_model['m/z'] = df_hits['m/z'].copy(), df_nonhits_KMD['m/z'].copy()
        for ref in refs_comb:
            df_hits_model[ref] = df_hits_KMD[ref].copy()
            df_nonhits_model[ref] = df_nonhits_KMD[ref].copy()

        df_hits_model['Label'] = np.full(shape=df_hits.shape[0], fill_value=1)
        df_nonhits_model['Label'] = np.full(shape=df_nonhits.shape[0], fill_value=0)

        df_train = pd.concat([df_hits_model, df_nonhits_model], axis=0)

        # Train random forest model, perform cross-validation
        X_train, y_train = df_train.iloc[:, :-1].values, df_train.iloc[:, -1].values
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1, random_state=0)

        t_start_CV = datetime.now()
        model.fit(X_train, y_train)

        # Perform cross-validation and permutation testing
        cv_scores = cross_val_score(estimator=model, X=X_train, y=y_train, cv=CV_fold, n_jobs=-1)
        t_end_CV = datetime.now()
        np.put(times_arr, ind=j, v=(t_end_CV-t_start_CV).total_seconds())
        for k, val in enumerate(cv_scores):
            cv_scores_arr[j, k] = val
        # print(datetime.now().strftime("%H:%M:%S") + ' | Finished cross-validation')

        if permutation_test:
            score, perm_scores, perm_pval = permutation_test_score(model, X_train, y_train, n_permutations=N_perm, cv=CV_fold, n_jobs=-1)
            for k, val in enumerate(perm_scores):
                perm_scores_arr[j, k] = val
            np.put(perm_score_arr, ind=j, v=score)
            np.put(perm_pvals, ind=j, v=perm_pval)
            print(datetime.now().strftime("%H:%M:%S") + ' | Finished permutation test')

    # Store results in dataframe, save to xlsx file
    for j in range(N_var):
        df_res['Var '+str(j + 1)] = arr_vars[:, j]
    df_cv_scores = pd.DataFrame(data=cv_scores_arr)
    df_res_CV = pd.concat([df_res, df_cv_scores], axis=1)
    df_res_CV.to_csv(fileloc_out+filename_out + str(max_depth)+' (CV scores, '+str(N_var)+' vars).csv', index=False)
    print(df_res_CV)

    df_res_times = df_res.copy()
    df_res_times['Duration [s]'] = times_arr
    df_res_times.to_csv(fileloc_out+filename_out + str(max_depth)+' (CV scores, '+str(N_var)+' vars, duration).csv', index=False)
    print(df_res_times)

    if permutation_test:
        df_perm_scores = pd.DataFrame(data=perm_scores_arr)
        df_res_perm = pd.concat([df_res, df_perm_scores], axis=1)
        df_res_perm['Permutation p-value'] = perm_pvals
        df_res_perm['Permutation score'] = perm_score_arr
        df_res_perm.to_csv(fileloc_out+filename_out + ' (permutation scores, '+str(N_var)+' vars).csv', index=False)
        print(df_res_perm)
