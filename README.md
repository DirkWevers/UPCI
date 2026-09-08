# UPCI
For the processing of our single-cell HRMS data, we have developed a collective of Python functions that together generate an untargeted peak confidence index (UPCI) that can be used to filter out noise from HRMS data. We have generated the following functions (more details per function are given in the code of the respective functions). Furthermore, the data generated using this code is shared via Zenodo (10.5281/zenodo.21536682). 

**CalcDataframeSizes** calculates the sizes of the dataframes that contain hits and non-hits, to help the user balance the training data set. It returns the number of rows to be sampled from the dataframe with the hits and the number of rows to be sampled from the dataframe with the non-hits.

**CalcMassDefect** calculates the mass defect of the input m/z values using a user-defined reference mass and adduct mass.

**ExtractScanPropertiesRawFile** extracts per scan in a raw file properties like polarity, FAIMS compensation voltage, m/z range, etc. Using this function, the user can select which scans in the raw file they want to process simultaneously. 

**SearchDatabase** is used to screen a user-input database for all the m/z values that are provided.

**UPCI_blacklist** removes the m/z values from the input dataframe that are within "tolerance" ppm of the m/z values input as "mz_blacklist". It gives two dataframes as output: one listing the m/z values that are kept, and one listing those that are removed.
