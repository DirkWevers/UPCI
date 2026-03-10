# UPCI
For the processing of our single-cell HRMS data, we have developed a collective of Python functions that together generate an untargeted peak confidence index (UPCI) that can be used to filter out noise from HRMS data. We have generated the following functions (more details per function are given in the code of the respective functions)

**UPCI_blacklist** removes the m/z values from the input dataframe that are within "tolerance" ppm of the m/z values input as "mz_blacklist". It gives two dataframes as output: one listing the m/z values that are kept, and one listing those that are removed.

**UPCI_whitelist** separates the input dataframe into two dataframes: one that contains the m/z values that are within "tolerance" ppm of the m/z values input as "mz_whitelist", and one that contains the other m/z values. This function can, for instance, be used to set apart the LipidMaps hits from the non-hits.

