import numpy as np
from math import factorial
import matplotlib.pylab as plt

def CST_2_coord(CST_parameter, dzu, dzl, N1=0.5, N2=1, x=None):

    if x is None:
        len_x = 69
        theta = np.linspace(0, np.pi, int(np.ceil(len_x / 2)))
        x0 = 0.5 * (np.cos(theta) + 1)
        x = np.concatenate((x0, x0[::-1][1:]))


    indices_le = np.where(x == 0)[0].item()
    xu = x[:indices_le+1]
    xl = x[indices_le:]

    wu = CST_parameter[:int(len(CST_parameter)/2)]
    wl = CST_parameter[int(len(CST_parameter)/2):]

    yu = class_shape_y(wu, xu, N1, N2, dzu)
    yl = class_shape_y(wl, xl, N1, N2, dzl)

    y = np.concatenate((yu, yl[1:]))

    coord = np.column_stack((x, y))
    return coord


def class_shape_y(w, x, N1, N2, dz):

    x[x < 0] = 0
    C = x ** N1 * ((1 - x) ** N2)
    n = len(w) - 1
    K = np.zeros(n + 1)
    for i in range(0, n + 1):
        K[i] = factorial(n) / (factorial(i) * (factorial((n - i))))

    S = np.zeros(len(x))
    for i in range(len(x)):
        for j in range(0, n + 1):
            S[i] += w[j] * K[j] * x[i] ** (j) * ((1 - x[i]) ** (n - j))

    # Calculate y output
    y = np.zeros(len(x))
    for i in range(len(y)):
        y[i] = C[i] * S[i] + x[i] * dz

    return y