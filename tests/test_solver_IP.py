import numpy as np
import sys
sys.path.append('./')
from solvers.integer_optimizers import complementary_products

V_mx = np.array([
    [10, 20, 30, 40, 50, 60],
    [10, 20, 30, 40, 50, 60],
    [30, 50, 30, 40, 50, 60],
    [10, 20, 30, 40, 80, 60],
    [10, 20, 30, 40, 50, 60],
    [10, 20, 30, 40, 50, 60]
])

x_vals, y_vals = complementary_products(V_mx, verbose=True)
