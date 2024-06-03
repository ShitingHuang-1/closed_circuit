import numpy as np
import scipy as sp
import sympy
import math
class sensitivity:
    def __init__(self):
        pass
    def sens_g(self,x, y):
        results = []
        for i in range(len(x)-1):
            for j in range(len(y)-1):
                delta_y = y[j+1] - y[j]
                delta_x = x[i+1] - x[i]
                dydx = delta_y/delta_x
                yx = y[j] / x[i]
                sen = 100 * dydx/yx
                if not isinstance(sen, np.ndarray):
                    v_sum = sen
                    results.append(v_sum)
                else:
                    v_sum = sum(sen)
                    results.append(v_sum/len(sen))
        update = np.array(results)
        results = sum(update)/(len(x)-1)
        return results