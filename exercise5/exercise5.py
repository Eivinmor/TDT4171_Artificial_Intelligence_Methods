import numpy as np
import matplotlib.pyplot as plt


r = np.arange(-6, 6.0001, 0.05)
w1_matrix, w2_matrix = np.meshgrid(r, r)


def delta(w, x):
    return 1 / (1 + np.exp(- np.inner(w, x)))


def L_simple(w):
    p1 = (delta(w, [1, 0]) - 1) ** 2
    p2 = (delta(w, [0, 1])) ** 2
    p3 = (delta(w, [1, 1]) - 1) ** 2
    return p1 + p2 + p3

values = [[None for i in range(len(r))] for i in range(len(r))]
for i in range(len(r)):
    for j in range(len(r)):
        w = [w1_matrix[i][j], w2_matrix[i][j]]
        values[i][j] = L_simple(w)


fig = plt.figure()
plt.plot()
plt.pcolormesh(w1_matrix, w2_matrix, values, cmap='RdBu')
plt.colorbar()
plt.show()
