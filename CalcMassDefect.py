import numpy as np

def calc_mass_defect(arr, ref_mass=0, adduct_mass=1.00727647):
    arr -= adduct_mass
    arr = np.array(arr, dtype='float64')
    if ref_mass != 0:
        exact_mass_arr = arr * np.round(ref_mass) / ref_mass    # Kendrick mass
        nom_mass_arr = np.round(exact_mass_arr)                 # Nominal Kendrick mass
        arr_MD = nom_mass_arr - exact_mass_arr                  # Kendrick mass defect
    elif ref_mass == 0:
        nom_mass_arr = np.round(arr)
        arr_MD = arr - nom_mass_arr
    return arr_MD
