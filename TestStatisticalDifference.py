# Import relevant functions
from scipy.stats import shapiro, levene, bartlett, ttest_ind
import numpy as np

def test_difference_stats(arr_a, arr_b, alternative='two-sided', alpha=0.05):
    # Check whether there are sufficient values to run a Shapiro-Wilk test
    if len(arr_a) >= 3 and len(arr_b) >= 3:
        # Run Shapiro-Wilk test on both data arrays
        SW_a = shapiro(arr_a)
        SW_b = shapiro(arr_b)
        SW_a_pval = SW_a.pvalue
        SW_b_pval = SW_b.pvalue

        # Test for equal variance if the SW-test is not significant
        if SW_a_pval > alpha and SW_b_pval > alpha:
            # Run Levene test if SW-test is not significant
            levene_pval = levene(arr_a, arr_b).pvalue
            bartlett_pval = np.nan
        else:
            # Run Bartlett's test if one or both SW-tests are significant
            bartlett_pval = bartlett(arr_a, arr_b).pvalue
            levene_pval = np.nan

        # Test for identical average values
        if levene_pval > alpha or bartlett_pval > alpha:     # variances are equal
            arr_a, arr_b = [float(el) for el in arr_a], [float(el) for el in arr_b]
            t_stat, ttest_pval = ttest_ind(arr_a, arr_b, equal_var=True, alternative=alternative)
            welch_pval = np.nan
        else:                   # variances are not equal
            arr_a, arr_b = [float(el) for el in arr_a], [float(el) for el in arr_b]
            t_stat, welch_pval = ttest_ind(arr_a, arr_b, equal_var=False, alternative=alternative)
            ttest_pval = np.nan
    else:
        print('Insufficient values for Shapiro-Wilk test')
        SW_a_pval, SW_b_pval, levene_pval, bartlett_pval, ttest_pval, welch_pval = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    return SW_a_pval, SW_b_pval, levene_pval, bartlett_pval, ttest_pval, welch_pval