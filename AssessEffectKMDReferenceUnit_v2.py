import pandas as pd
import numpy as np
import statsmodels.api as sm

def assess_statistical_effect(df_in, N_vars):
    df_CV = df_in.iloc[:, -10:]
    df_in_vars = df_in.iloc[:, :N_vars]
    median_CV_arr = np.array(df_CV.median(axis=1))

    ref_names = np.unique(np.array(df_in_vars))
    df_dummy = pd.DataFrame(columns=ref_names, data=np.zeros(shape=(df_in.shape[0], len(ref_names))))

    for i in range(df_in.shape[0]):
        df_in_row = df_in_vars.iloc[i, :]
        row_vars = np.array(df_in_row)
        for var in row_vars:
            df_dummy.at[i, var] = 1

    # Perform linear regression
    df_dummy = sm.add_constant(df_dummy)
    model = sm.OLS(median_CV_arr, df_dummy)
    res = model.fit()
    return res

def results_to_DF(res, alpha=0.05):
    df = pd.DataFrame(columns=['Ref. unit', 'Coefficient and CI', 'p-value'], dtype='object')
    df['Ref. unit'] = res.params.index

    coefs = res.params.values
    CI_LL = res.conf_int(alpha=alpha).values[:, 0]
    coefs_CIs = []
    for coef, CI_L in zip(coefs, CI_LL):
        coefs_CIs.append(str(round(coef, 3)) + ' \u00B1 ' + str(round(coef - CI_L, 3)))

    df['Coefficient and CI'] = coefs_CIs

    df['p-value'] = res.pvalues.values
    return df

# Adjustable variables
fileloc = 'C:\\Users\\dirkw\\PycharmProjects\\pythonProject\\PhD\\UPCI\\OutputData\\'
df_depth5_2vars = pd.read_csv(fileloc+'202605 RF performance for multiple KMDs, v4, max depth 5 (CV scores, 2 vars).csv')
res_depth5_2vars = assess_statistical_effect(df_depth5_2vars, 2)
df_depth10_2vars = pd.read_csv(fileloc+'202605 RF performance for multiple KMDs, v4, max depth 10 (CV scores, 2 vars).csv')
res_depth10_2vars = assess_statistical_effect(df_depth10_2vars, 2)
df_depth15_2vars = pd.read_csv(fileloc+'202605 RF performance for multiple KMDs, v4, max depth 15 (CV scores, 2 vars).csv')
res_depth15_2vars = assess_statistical_effect(df_depth15_2vars, 2)
df_depth20_2vars = pd.read_csv(fileloc+'202605 RF performance for multiple KMDs, v4, max depth 20 (CV scores, 2 vars).csv')
res_depth20_2vars = assess_statistical_effect(df_depth20_2vars, 2)
df_depth30_2vars = pd.read_csv(fileloc+'202605 RF performance for multiple KMDs, v4, max depth 30 (CV scores, 2 vars).csv')
res_depth30_2vars = assess_statistical_effect(df_depth30_2vars, 2)

df_depth5_3vars = pd.read_csv(fileloc+'202605 RF performance for multiple KMDs, v4, max depth 5 (CV scores, 3 vars).csv')
res_depth5_3vars = assess_statistical_effect(df_depth5_3vars, 3)
df_depth10_3vars = pd.read_csv(fileloc+'202605 RF performance for multiple KMDs, v4, max depth 10 (CV scores, 3 vars).csv')
res_depth10_3vars = assess_statistical_effect(df_depth10_3vars, 3)
df_depth15_3vars = pd.read_csv(fileloc+'202605 RF performance for multiple KMDs, v4, max depth 15 (CV scores, 3 vars).csv')
res_depth15_3vars = assess_statistical_effect(df_depth15_3vars, 3)
df_depth20_3vars = pd.read_csv(fileloc+'202605 RF performance for multiple KMDs, v4, max depth 20 (CV scores, 3 vars).csv')
res_depth20_3vars = assess_statistical_effect(df_depth20_3vars, 3)
df_depth30_3vars = pd.read_csv(fileloc+'202605 RF performance for multiple KMDs, v4, max depth 30 (CV scores, 3 vars).csv')
res_depth30_3vars = assess_statistical_effect(df_depth30_3vars, 3)

# Generate writer to save output
writer = pd.ExcelWriter(fileloc+'20260820 UPCI KMD effect - Linear regression results.xlsx', engine='openpyxl')

print('Max depth = 5, 2 variables')
print(res_depth5_2vars.summary())
df_res = results_to_DF(res_depth5_2vars)
df_res.to_excel(writer, index=False, sheet_name='Max depth = 5, 2 vars')
print('Max depth = 10, 2 variables')
print(res_depth10_2vars.summary())
df_res = results_to_DF(res_depth10_2vars)
df_res.to_excel(writer, index=False, sheet_name='Max depth = 10, 2 vars')
print('Max depth = 15, 2 variables')
print(res_depth15_2vars.summary())
df_res = results_to_DF(res_depth15_2vars)
df_res.to_excel(writer, index=False, sheet_name='Max depth = 15, 2 vars')
print('Max depth = 20, 2 variables')
print(res_depth20_2vars.summary())
df_res = results_to_DF(res_depth20_2vars)
df_res.to_excel(writer, index=False, sheet_name='Max depth = 20, 2 vars')
print('Max depth = 30, 2 variables')
print(res_depth30_2vars.summary())
df_res = results_to_DF(res_depth30_2vars)
df_res.to_excel(writer, index=False, sheet_name='Max depth = 30, 2 vars')

print('Max depth = 5, 3 variables')
print(res_depth5_3vars.summary())
df_res = results_to_DF(res_depth5_3vars)
df_res.to_excel(writer, index=False, sheet_name='Max depth = 5, 3 vars')
print('Max depth = 10, 3 variables')
print(res_depth10_3vars.summary())
df_res = results_to_DF(res_depth10_3vars)
df_res.to_excel(writer, index=False, sheet_name='Max depth = 10, 3 vars')
print('Max depth = 15, 3 variables')
print(res_depth15_3vars.summary())
df_res = results_to_DF(res_depth15_3vars)
df_res.to_excel(writer, index=False, sheet_name='Max depth = 15, 3 vars')
print('Max depth = 20, 3 variables')
print(res_depth20_3vars.summary())
df_res = results_to_DF(res_depth20_3vars)
df_res.to_excel(writer, index=False, sheet_name='Max depth = 20, 3 vars')
print('Max depth = 30, 3 variables')
print(res_depth30_3vars.summary())
df_res = results_to_DF(res_depth30_3vars)
df_res.to_excel(writer, index=False, sheet_name='Max depth = 30, 3 vars')

writer.close()