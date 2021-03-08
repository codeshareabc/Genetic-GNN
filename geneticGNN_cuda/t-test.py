import numpy as np
import math
from scipy.stats import t

def significant_test(a_m, b_m, a_s, b_s, n, alpha):
    mean_diff = a_m - b_m
    print("mean_diff",mean_diff)
    mean_diff_s = np.sqrt(a_s*a_s/n + b_s*b_s/n)
    print("mean_diff_s",mean_diff_s)
    deg_freedom = ((a_s*a_s/n + b_s*b_s/n)**2) / (1/(n+1)*math.pow(a_s*a_s/n, 2)+1/(n+1)*math.pow(b_s*b_s/n,2)) - 2
    print("deg_freedom",deg_freedom)
    vals = t.ppf(alpha, deg_freedom) * mean_diff_s
    print(vals)
    low_bound = mean_diff - vals
    high_bound = mean_diff + vals
    print(low_bound, high_bound)
    if low_bound*high_bound < 0:
        return 0
    return 1



result = significant_test(0.578, 0.564, 0.012, 0.011, 10, 0.95)

print(result)