import pandas as pd
import numpy as np
from numpy.random import seed

def calc_dataframe_sizes(N, N_noise_max, balance_ratio):  # TO DO: must always return an integer value
    if balance_ratio < 1:
        N_noise = balance_ratio * N
        N_hits = N
    elif balance_ratio == 1:
        N_noise = N
        N_hits = N
    elif balance_ratio > 1:
        N_noise = balance_ratio * N
        N_hits = N
        if N_noise > N_noise_max:
            N_noise = N_noise_max
            N_hits = int(round(N_noise_max / balance_ratio))
    return N_hits, N_noise

