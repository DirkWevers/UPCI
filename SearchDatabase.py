import pandas as pd
import numpy as np
from datetime import datetime

def search_database(db_file, df_data, data_col_mz, DB_col_mz, col_comp, col_ID, ppm_threshold=5, info=True):
    # Generate empty variables
    df_cols = ['m/z screen', 'm/z hit', 'El. comp', 'ID', 'ppm error']
    df_hits = pd.DataFrame(columns=df_cols)
    ind_nonhits = []

    # Extract m/z values from data
    mz_screen = np.array(df_data[data_col_mz])

    t_start = datetime.now()
    for i in range(len(mz_screen)):
        mz = mz_screen[i]
        if info and (i + 1) % 1000 == 0:
            ETF = (datetime.now() - t_start) / (i + 1) * len(mz_screen)
            print(datetime.now().strftime("%H:%M:%S") + ' | Started processing m/z value ' + str(i + 1) + ' of ' + str(len(mz_screen)) + ' | ETF: ' + (t_start + ETF).strftime("%d %B, %H:%M:%S"))
        mz_LL, mz_UL = mz*(1 - ppm_threshold*10**-6), mz*(1 + ppm_threshold*10**-6)
        df_check = db_file[np.logical_and(db_file[DB_col_mz] > mz_LL, db_file[DB_col_mz] < mz_UL)]
        if not df_check.empty:
            df_hit = pd.DataFrame(columns=df_cols, data=np.zeros(shape=(df_check.shape[0], len(df_cols))))
            df_hit['m/z screen'] = np.full(shape=df_check.shape[0], fill_value=mz)
            df_hit['m/z hit'] = np.array(df_check[DB_col_mz])
            df_hit['El. comp'] = np.array(df_check[col_comp])
            df_hit['ID'] = np.array(df_check[col_ID])
            ppm_errors = (np.array(df_check['m/z']) - mz)/mz * 10**6
            df_hit['ppm error'] = ppm_errors
            df_hits = pd.concat([df_hits, df_hit], axis=0)
        else:
            ind_nonhits += [i]

    # Process dataframe with hits
    df_hits.sort_values(by=['m/z screen', 'ppm error'], ascending=True)

    # Process dataframe with non-hits
    df_nonhits = df_data.loc[ind_nonhits, :]
    df_nonhits.sort_values(by=data_col_mz, inplace=True, ascending=True)

    return df_hits, df_nonhits

# df_hits = search_database(df_database, screen_mz, 'm/z', 'Isotopic composition', 'Isotopic composition')
# df_hits.to_excel('20251118 0p25cells-uL_01 - GCTAn results (HMDB hits, 10E6 int).xlsx')
