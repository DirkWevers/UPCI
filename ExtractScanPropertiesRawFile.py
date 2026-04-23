import pandas as pd
import numpy as np
from fisher_py import RawFile

def extract_scan_properties(raw_file):
    N_scans = raw_file.number_of_scans

    # Extract scan properties
    df_scanprops = pd.DataFrame(columns=['Scan nr.', 'Polarity', 'FAIMS CV', 'm/z range', 'm/z start', 'm/z end', 'MS2 central m/z'])
    df_scanprops['Scan nr.'] = np.arange(1, N_scans+1)
    polarity_list, FAIMS_CV_arr, mz_range_list, mz_start_arr, mz_end_arr, central_mz = [], np.full(shape=N_scans, fill_value=0), [], np.full(shape=N_scans, fill_value=np.nan), np.full(shape=N_scans, fill_value=np.nan), np.full(shape=N_scans, fill_value=np.nan)
    for scan_nr in range(1, N_scans+1):
        scan_description = raw_file.get_scan_from_scan_number(scan_nr)[-1]

        # Polarity
        if ' + p' in scan_description:
            polarity_list += ['+']
        elif ' - p' in scan_description:
            polarity_list += ['-']

        # FAIMS CV
        if 'cv=' in scan_description:
            scan_description_split = scan_description.split('cv=')
            scan_description_split = scan_description_split[1].split(' ')
            CV = float(scan_description_split[0])
            np.put(a=FAIMS_CV_arr, v=CV, ind=scan_nr - 1)

        # m/z range
        scan_description_split = scan_description.split('[')
        mz_range = scan_description_split[1][:-1]
        mz_range_list += [mz_range]
        np.put(a=mz_start_arr, ind=scan_nr - 1, v=float(mz_range.split('-')[0]))
        np.put(a=mz_end_arr, ind=scan_nr - 1, v=float(mz_range.split('-')[1]))

        # MS/MS
        if 'ms2' in scan_description:
            scan_description_split = scan_description.split('ms2')[1]
            scan_description_split = scan_description_split.split('@')[0]
            np.put(a=central_mz, ind=scan_nr - 1, v=float(scan_description_split))

    # Store output
    df_scanprops['Polarity'] = polarity_list
    df_scanprops['FAIMS CV'] = FAIMS_CV_arr
    df_scanprops['m/z range'] = mz_range_list
    df_scanprops['m/z start'] = mz_start_arr
    df_scanprops['m/z end'] = mz_end_arr
    df_scanprops['MS2 central m/z'] = central_mz

    return df_scanprops

# raw_file = RawFile('C:\\Users\\dirkw\\PycharmProjects\\pythonProject\\PhD\\Data\\20260402_Test_Full-DIA\\20260402_StdMix_Full-DIA.raw')
# df_scanprops = extract_scan_properties(raw_file)
# print(df_scanprops)
