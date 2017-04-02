import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

r = np.arange(-6, 7, 1)
x_m, y_m = np.meshgrid(r, r)



def delta(w, x):
    return 1 / (1 + np.exp(- np.inner(w, x)))


def L_simple(w):
    p1 = (delta(w, [1, 0]) - 1) ** 2
    p2 = (delta(w, [0, 1])) ** 2
    p3 = (delta(w, [1, 1]) - 1) ** 2
    return p1 + p2 + p3

values = []

for i in range(-6, 7):
    row = []
    for j in range(-6, 7):
        row.append(L_simple([x_m[i][j], y_m[i][j]]))
    values.append(row)

values = np.matrix(values)
print(values)


fig = plt.figure()
ax = fig.gca(projection="3d")
surf = ax.plot_surface(x_m, y_m, values, cmap=cm.coolwarm, linewidth=0, antialiased=True)


# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()

# plt.plot(x_m, y_m, marker=".", color="k", linestyle="none")
# plt.show()