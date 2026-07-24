import pandas as pd
import numpy as np
from numpy.random import seed
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.naive_bayes import GaussianNB
from UPCIFunctions.CalcMassDefect import calc_mass_defect
from UPCIFunctions.CalcDataframeSizes import calc_dataframe_sizes
from datetime import datetime
from itertools import product

df_pars = pd.read_excel('C:\\Users\\dirkw\\PycharmProjects\\pythonProject\\PhD\\UPCI\\HyperparameterOptimization\\HyperParOpt_Combined_InputPars.xlsx')

for i in range(df_pars.shape[0]):
    df_row = df_pars.iloc[i, :]

    # Adjustable parameters
    N_iter = df_row['N_iter']
    CV_fold = df_row['CV_fold']
    ref_masses = df_row['ref_masses']
    balance_ratio = df_row['balance_ratio']
    N_hits_max = df_row['N_hits_max']

    if isinstance(ref_masses, str):
        ref_masses = [float(el.strip()) for el in ref_masses.split(';')]
    elif np.isnan(ref_masses):
        ref_masses = []
    elif isinstance(ref_masses, float):
        ref_masses = [ref_masses]

    # Loading in data into python
    fileloc = df_row['fileloc']
    filename_hits = df_row['filename_hits']
    filename_nonhits = df_row['filename_nonhits']
    fileloc_out = df_row['fileloc_out']
    filename_out = df_row['filename_out']
    model_name = df_row['Model']

    # Load and select training data, calculate KMD values
    df_hits = pd.read_csv(fileloc + filename_hits)
    df_nonhits = pd.read_csv(fileloc + filename_nonhits)

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

    # Define parameter grid
    if model_name == 'RF':
        N_est = [int(round(el)) for el in np.logspace(start=np.log10(50), stop=np.log10(500), num=40)]
        max_depth = [int(round(el)) for el in np.linspace(start=3, stop=40, num=15)]
        min_samples_split = [int(round(el)) for el in np.linspace(start=2, stop=40, num=15)]
        max_features = ['sqrt', 'log2']

        list_lists = [N_est, max_depth, min_samples_split, max_features]
        combs = list(product(*list_lists))
        df_parcombs = pd.DataFrame(data=combs, columns=['N_est', 'max_depth', 'min_samples_split', 'max_features'])
        df_parcombs_sel = df_parcombs.sample(N_iter, replace=False, ignore_index=True)

        # Define dataframe for output
        df_res_cols = ['N_estimators', 'max_depth', 'min_samples_split', 'max_features', 'median score', 'mean score', 'stdev score', 'Duration']
        df_res = pd.DataFrame(dtype='object', columns=df_res_cols, data=np.zeros(shape=(N_iter, len(df_res_cols))))

    elif model_name == 'GBT':
        learning_rate = list(np.linspace(start=0.01, stop=.5, num=20))
        N_est = [int(round(el)) for el in np.logspace(start=10, stop=np.log10(1000), num=30)]
        max_depth = [int(round(el)) for el in np.linspace(start=3, stop=20, num=12)]
        subsamples = list(np.linspace(start=0.25, stop=1, num=15))
        max_features = ['sqrt', 'log2']
        min_samples_split = list(range(2, 6))
        min_samples_leaf = list(range(1, 6))

        list_lists = [learning_rate, N_est, max_depth, subsamples, min_samples_split, max_features, min_samples_leaf]
        combs = list(product(*list_lists))
        df_parcombs = pd.DataFrame(data=combs, columns=['learning rate', 'N_est', 'max_depth', 'subsamples', 'min_samples_split', 'max_features', 'min_samples_leaf'])
        df_parcombs_sel = df_parcombs.sample(N_iter, replace=False, ignore_index=True)

        # Define dataframe for output
        df_res_cols = ['learning rate', 'N_est', 'max_depth', 'subsamples', 'min_samples_split', 'max_features', 'min_samples_leaf', 'median score', 'mean score', 'stdev score', 'Duration']
        df_res = pd.DataFrame(dtype='object', columns=df_res_cols, data=np.zeros(shape=(N_iter, len(df_res_cols))))

    elif model_name == 'LGBM':
        # Define parameter grid
        learning_rate = list(np.linspace(start=0.01, stop=1, num=20))
        N_est = [int(round(el)) for el in np.logspace(start=1, stop=np.log10(500), num=30)]
        N_leaves = [int(round(el)) for el in np.logspace(start=3, stop=8, num=30, base=2)]
        max_depth = [int(round(el)) for el in np.linspace(start=3, stop=40, num=15)]
        min_samples_leaf = list(range(1, 6))

        list_lists = [learning_rate, N_est, N_leaves, max_depth, min_samples_leaf]
        combs = list(product(*list_lists))
        df_parcombs = pd.DataFrame(data=combs, columns=['learning rate', 'N_est', 'N_leaves', 'max_depth', 'min_samples_leaf'])
        df_parcombs = df_parcombs[df_parcombs['N_leaves'] <= 2 ** df_parcombs['max_depth']]
        df_parcombs_sel = df_parcombs.sample(N_iter, replace=False, ignore_index=True)

        # Define dataframe for output
        df_res_cols = ['N_estimators', 'max_depth', 'learning rate', 'min_samples_split', 'N_leaves', 'median score', 'mean score', 'stdev score', 'Duration']
        df_res = pd.DataFrame(dtype='object', columns=df_res_cols, data=np.zeros(shape=(N_iter, len(df_res_cols))))

    elif model_name == 'LR':
        C = list(np.logspace(start=-4, stop=3, num=100))
        l1_ratio = np.linspace(start=0, stop=1, num=100)

        list_lists = [C, l1_ratio]
        combs = list(product(*list_lists))
        df_parcombs = pd.DataFrame(data=combs, columns=['C', 'l1_ratio'])
        df_parcombs_sel = df_parcombs.sample(N_iter, replace=False, ignore_index=True)

        # Define dataframe for output
        df_res_cols = ['C', 'l1_ratio', 'median score', 'mean score', 'stdev score', 'Duration']
        df_res = pd.DataFrame(dtype='object', columns=df_res_cols, data=np.zeros(shape=(N_iter, len(df_res_cols))))

    elif model_name == 'NB':
        # Define parameter grid for NB model
        var_smoothing = list(np.logspace(start=-13, stop=-5, num=N_iter))
        df_parcombs = pd.DataFrame(data=var_smoothing, columns=['var_smoothing'])
        df_parcombs_sel = df_parcombs.sample(N_iter, replace=False, ignore_index=True)

        # Define dataframe for output
        df_res_cols = ['var_smoothing', 'median score', 'mean score', 'stdev score', 'Duration', 'CV scores']
        df_res = pd.DataFrame(dtype='object', columns=df_res_cols, data=np.zeros(shape=(N_iter, len(df_res_cols))))

    elif model_name == 'SVM':
        # Define parameter grid
        gamma = list(np.logspace(-3, 0, 25))
        C = list(np.logspace(-5, 1, 50))
        kernel = ['linear', 'rbf']

        list_lists = [gamma, C, kernel]
        combs = list(product(*list_lists))
        df_parcombs = pd.DataFrame(data=combs, columns=['gamma', 'C', 'kernel'])
        df_parcombs['gamma'][df_parcombs['kernel'] == 'linear'] = 'scale'
        df_parcombs.drop_duplicates(inplace=True)
        df_parcombs_sel = df_parcombs.sample(N_iter, replace=False, ignore_index=True)

        # Define dataframe for output
        df_res_cols = ['gamma', 'C', 'degree', 'median score', 'mean score', 'stdev score', 'Duration']
        df_res = pd.DataFrame(dtype='object', columns=df_res_cols, data=np.zeros(shape=(N_iter, len(df_res_cols))))

    elif model_name == 'XGB':
        # Define parameter grid
        learning_rate = list(np.logspace(start=-4, stop=np.log10(.5), num=30))
        N_est = [int(round(el)) for el in np.logspace(start=np.log10(50), stop=np.log10(5000), num=30)]
        max_depth = [int(round(el)) for el in np.linspace(start=3, stop=20, num=10)]
        subsamples = list(np.linspace(start=0.25, stop=1, num=10))
        min_child_weight = list(np.logspace(start=0, stop=np.log10(30), num=15))
        gamma = list(np.logspace(start=-2, stop=np.log10(20), num=30))

        list_lists = [learning_rate, N_est, max_depth, subsamples, min_child_weight, gamma]
        combs = list(product(*list_lists))
        df_parcombs = pd.DataFrame(data=combs, columns=['learning rate', 'N_est', 'max_depth', 'subsamples', 'min_child_weight', 'gamma'])
        df_parcombs_sel = df_parcombs.sample(N_iter, replace=False, ignore_index=True)

        # Define dataframe for output
        df_res_cols = ['learning rate', 'N_est', 'max_depth', 'subsamples', 'min_child_weight', 'gamma', 'median score', 'mean score', 'stdev score', 'Duration']
        df_res = pd.DataFrame(dtype='object', columns=df_res_cols, data=np.zeros(shape=(N_iter, len(df_res_cols))))

    t_start_analyses = datetime.now()

    for iter in list(range(df_parcombs_sel.shape[0])):
        if iter > 0:
            ETF_analyses = (datetime.now() - t_start_analyses) / iter * N_iter
            print(datetime.now().strftime("%H:%M:%S") + ' | Processing analysis', iter + 1, 'of', N_iter, '| ETF: ' + (t_start_analyses + ETF_analyses).strftime("%d %B, %H:%M:%S"))
        else:
            print(datetime.now().strftime("%H:%M:%S") + ' | Processing analysis', iter + 1, 'of', N_iter)
        seed(iter)

        # Select data
        N_hits, N_noise = calc_dataframe_sizes(N=N_hits_max, N_noise_max=df_nonhits.shape[0], balance_ratio=balance_ratio)
        df_nonhits_sel = df_nonhits.sample(N_noise, replace=False, ignore_index=True)
        df_train = pd.concat([df_train_hits, df_train_nonhits], axis=0)
        X_train, y_train = df_train.iloc[:, :-1].values, df_train.iloc[:, -1].values
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1, random_state=0)
        print(datetime.now().strftime("%H:%M:%S") + ' | Finished loading data set')

        # Select parameter values
        df_row = df_parcombs_sel.iloc[iter, :]
        if model_name == 'RF':
            N_est_sel = df_row['N_est']
            max_depth_sel = df_row['max_depth']
            min_samples_sel = df_row['min_samples_split']
            max_features_sel = df_row['max_features']

            # Fit model
            model = RandomForestClassifier(n_estimators=N_est_sel, max_depth=max_depth_sel, min_samples_split=min_samples_sel, max_features=max_features_sel, random_state=iter)
            print(datetime.now().strftime("%H:%M:%S") + ' | Finished generating RF model')

            # Save run parameters
            df_res.loc[iter, 'N_estimators'] = N_est_sel
            df_res.loc[iter, 'max_depth'] = max_depth_sel
            df_res.loc[iter, 'min_samples_split'] = min_samples_sel
            df_res.loc[iter, 'max_features'] = max_features_sel

        elif model_name == 'GBT':
            # Select parameter values
            df_row = df_parcombs_sel.iloc[iter, :]
            learning_rate_sel = df_row['learning rate']
            N_est_sel = df_row['N_est']
            max_depth_sel = df_row['max_depth']
            min_samples_split_sel = df_row['min_samples_split']
            min_samples_leaf_sel = df_row['min_samples_leaf']
            max_features_sel = df_row['max_features']
            subsamples_sel = df_row['subsamples']

            # Fit model
            model = GradientBoostingClassifier(learning_rate=learning_rate_sel, n_estimators=N_est_sel, max_depth=max_depth_sel, min_samples_split=min_samples_split_sel, min_samples_leaf=min_samples_leaf_sel, max_features=max_features_sel, subsample=subsamples_sel, random_state=iter)
            print(datetime.now().strftime("%H:%M:%S") + ' | Finished generating GBT model')

            # Write output to dataframe
            df_res.loc[iter, 'learning rate'] = learning_rate_sel
            df_res.loc[iter, 'N_estimators'] = N_est_sel
            df_res.loc[iter, 'max_depth'] = max_depth_sel
            df_res.loc[iter, 'min_samples_split'] = min_samples_split_sel
            df_res.loc[iter, 'min_samples_leaf'] = min_samples_leaf_sel
            df_res.loc[iter, 'max_features'] = max_features_sel
            df_res.loc[iter, 'subsample'] = subsamples_sel

        elif model_name == 'LGBM':
            # Select parameter values
            df_row = df_parcombs_sel.iloc[iter, :]
            learning_rate_sel = df_row['learning rate']
            N_est_sel = int(df_row['N_est'])
            N_leaves_sel = int(df_row['N_leaves'])
            max_depth_sel = int(df_row['max_depth'])
            min_samples_leaf_sel = int(df_row['min_samples_leaf'])

            # Fit model
            model = LGBMClassifier(learning_rate=learning_rate_sel, num_leaves=N_leaves_sel, n_estimators=N_est_sel, max_depth=max_depth_sel, min_samples_leaf=min_samples_leaf_sel, random_state=iter, num_iterations=50)
            print(datetime.now().strftime("%H:%M:%S") + ' | Finished generating LGBM model')

            # Write output to dataframe
            df_res.loc[iter, 'learning rate'] = learning_rate_sel
            df_res.loc[iter, 'N_estimators'] = N_est_sel
            df_res.loc[iter, 'max_depth'] = max_depth_sel
            df_res.loc[iter, 'min_samples_leaf'] = min_samples_leaf_sel
            df_res.loc[iter, 'N_leaves'] = N_leaves_sel

        elif model_name == 'LR':
            # Select parameter values
            df_row = df_parcombs_sel.iloc[iter, :]
            C_sel = df_row['C']
            l1_ratio_sel = df_row['l1_ratio']
            print(C_sel, l1_ratio_sel)

            # Fit model
            model = LogisticRegression(l1_ratio=l1_ratio_sel, C=C_sel, random_state=iter, solver='saga', penalty='elasticnet')
            print(datetime.now().strftime("%H:%M:%S") + ' | Finished generating LR model')

            # Save run parameters
            df_res.at[iter, 'C'] = C_sel
            df_res.at[iter, 'l1_ratio'] = l1_ratio_sel

        elif model_name == 'NB':
            # Select parameter values
            df_row = df_parcombs_sel.iloc[iter, :]
            var_smoothing_sel = df_row['var_smoothing']

            # Fit model
            model = GaussianNB(var_smoothing=var_smoothing_sel)
            print(datetime.now().strftime("%H:%M:%S") + ' | Finished generating NB model')

            # Write output to dataframe
            df_res.loc[iter, 'var_smoothing'] = var_smoothing_sel

        elif model_name == 'SVM':
            # Select parameter values
            df_row = df_parcombs_sel.iloc[iter, :]
            gamma_sel = df_row['gamma']
            C_sel = df_row['C']
            kernel_sel = df_row['kernel']

            # Fit model
            model = SVC(kernel=kernel_sel, gamma=gamma_sel, C=C_sel)
            print(datetime.now().strftime("%H:%M:%S") + ' | Finished generating SVM model')

            # Write output to dataframe
            df_res.loc[iter, 'gamma'] = gamma_sel
            df_res.loc[iter, 'C'] = C_sel
            df_res.loc[iter, 'kernel'] = kernel_sel

        elif model_name == 'XGB':
            learning_rate_sel = df_row['learning rate']
            N_est_sel = int(df_row['N_est'])
            max_depth_sel = int(df_row['max_depth'])
            min_child_weight_sel = df_row['min_child_weight']
            subsamples_sel = df_row['subsamples']
            gamma_sel = df_row['gamma']

            # Fit model
            model = XGBClassifier(n_estimators=N_est_sel, max_depth=max_depth_sel, learning_rate=learning_rate_sel, min_child_weight=min_child_weight_sel, subsample=subsamples_sel, gamma=gamma_sel, random_state=iter)
            print(datetime.now().strftime("%H:%M:%S") + ' | Finished generating XGB model')

            # Write output to dataframe
            df_res.loc[iter, 'N_estimators'] = N_est_sel
            df_res.loc[iter, 'max_depth'] = max_depth_sel
            df_res.loc[iter, 'learning rate'] = learning_rate_sel
            df_res.loc[iter, 'min_child_weight'] = min_child_weight_sel
            df_res.loc[iter, 'subsample'] = subsamples_sel
            df_res.loc[iter, 'gamma'] = gamma_sel

        # Perform CV
        t_start = datetime.now()
        cv_scores = cross_val_score(estimator=model, X=X_train, y=y_train, cv=CV_fold, n_jobs=-1, verbose=10)
        t_end = datetime.now()

        # Save general output
        df_res.at[iter, 'median score'] = np.median(cv_scores)
        df_res.at[iter, 'mean score'] = np.mean(cv_scores)
        df_res.at[iter, 'stdev score'] = np.std(cv_scores)
        df_res.at[iter, 'Mean duration'] = (t_end - t_start).total_seconds() / CV_fold
        print(df_res)

        # Overwrite and save output
        df_res.to_csv(fileloc_out+filename_out, index=False)
