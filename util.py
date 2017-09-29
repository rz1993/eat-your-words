import math
import numpy as np

def sumexp(arr):
    return np.sum(np.exp(arr))

def logsumexp(arr):
    val = np.sum(np.exp(arr))
    if val == 0:
        return float('-1e300')
    return math.log(val)
