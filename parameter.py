import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

cdict = {'red':  ((0.0, 0.0, 0.0),
                 (1/6., 0.0, 0.0),
                 (1/2., 0.8, 1.0),
                 (5/6., 1.0, 1.0),
                 (1.0, 0.4, 1.0)),

             'green':  ((0.0, 0.0, 0.4),
                 (1/6., 1.0, 1.0),
                 (1/2., 1.0, 0.8),
                 (5/6., 0.0, 0.0),
                 (1.0, 0.0, 0.0)),

             'blue': ((0.0, 0.0, 0.0),
                 (1/6., 0.0, 0.0),
                 (1/2., 0.9, 0.9),
                 (5/6., 0.0, 0.0),
                 (1.0, 0.0, 0.0))

        }

cmap = LinearSegmentedColormap('rg',cdict, N=256)

def valid(na, nv, m, xmax, tmax):
    v = np.linspace(1, nv/9.549, 5000)
    a = np.linspace(1, na, 6000)
    v, a = np.meshgrid(v, a)
    t1 = a / m
    t2 = v / a - t1
    xtogo = xmax - 2 * (3 * m / 2 * t2 * t1 ** 2 + m / 2 * t1 * t2 ** 2 + m * t1 ** 3)
    t_const = xtogo / (m * t1 * t2 + m * t1 ** 2)
    return (v / a >= a / m) & (t_const > 0) & (2*t1 + 2*t2 + 2*t1 + t_const < tmax)


maxv = 5000
maxa = 6000
m = 100000    # in rad per cubic seconds
xmax = 105*80/180*np.pi
tmax = 0.5

grid = valid(maxa, maxv, m, xmax, tmax)
print(grid)


plt.imshow(~grid, cmap=cmap)
#plt.scatter(2000,2000)
plt.xlabel("v")
plt.ylabel("a")
plt.savefig("parameters2.pdf", transparent=True)
plt.show()