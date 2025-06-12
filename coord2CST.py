from CST2coord import *
from scipy import optimize

def coord_2_CST(coord, dzu, dzl, order=6):

    CST_parameter_0 = np.zeros(2 * (order + 1))
    CST_parameter_bounds = ((-2, 2),) * (2 * (order + 1))

    def function_CST(CST_parameter, dzu, dzl):
        coord2 = CST_2_coord(CST_parameter=CST_parameter, dzu=dzu, dzl=dzl, x=coord[:, 0])
        y2 = coord2[:, 1]
        f = 0
        for i in range(len(y2)):
            f += (coord[i, 1] * 100 - y2[i] * 100) ** 2
            #f是目标函数
        return f
    res = optimize.minimize(function_CST,  CST_parameter_0, args=(dzu, dzl), method='SLSQP',
                            options={'ftol': 1e-16, 'disp': False},
                            bounds=CST_parameter_bounds)

    return res.x, res.fun