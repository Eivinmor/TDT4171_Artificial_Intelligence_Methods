import numpy as np
import random
import matplotlib.pyplot as plt
import copy


r = np.arange(-6, 6.0001, 0.05)
w1_matrix, w2_matrix = np.meshgrid(r, r)


def delta(w, x):
    return 1 / (1 + np.exp(- np.inner(w, x)))


# def delta(x):
#     return 1 / (1 + np.exp(-x))
def L_simple(w):
    p1 = (delta(w, [1, 0]) - 1) ** 2
    p2 = (delta(w, [0, 1])) ** 2
    p3 = (delta(w, [1, 1]) - 1) ** 2
    return p1 + p2 + p3


# def L_simple(w):
#     p1 = (delta(w[0]) - 1) ** 2
#     p2 = (delta(w[1])) ** 2
#     p3 = (delta(w[0]+w[1]) - 1) ** 2
#     return p1 + p2 + p3
def updateWeights(w, learning_rate):
    w[0] += - learning_rate * (L_simple_deriv(w, 0))
    w[1] += - learning_rate * (L_simple_deriv(w, 1))


def L_simple_deriv(w, i):
    a = (delta(w, [1, 0]) - 1) * (delta(w, [1, 0])) * (1 - delta(w, [1, 0])) * [1, 0][i]
    b = (delta(w, [0, 1]))     * (delta(w, [0, 1])) * (1 - delta(w, [0, 1])) * [0, 1][i]
    c = (delta(w, [1, 1]) - 1) * (delta(w, [1, 1])) * (1 - delta(w, [1, 1])) * [1, 1][i]
    return a + b + c


values = [[None for i in range(len(r))] for i in range(len(r))]
for i in range(len(r)):
    for j in range(len(r)):
        w = [w1_matrix[i][j], w2_matrix[i][j]]
        values[i][j] = L_simple(w)


def gradient_descent(learning_rate, iterations, init_w):
    w = copy.copy(init_w)
    L_simple_storage = []
    for i in range(iterations):
        L_simple_storage.append(L_simple(w))
        updateWeights(w, learning_rate)
        if i % (iterations/20) == 0:
            print("|", end="", flush=True)
    print("\n")
    return L_simple_storage


learning_rates = [0.0001, 0.01, 0.1, 1, 10, 100]
iterations = 10000
# init_w = [r[random.randint(0, r.size - 1)], r[random.randint(0, r.size - 1)]]
init_w = [0, 6]

for lr in learning_rates:
    print("Running GD with lr =", lr)
    values = gradient_descent(lr, iterations, init_w)
    plt.plot(values, label=lr)
plt.legend(loc=1)
plt.show()

# fig = plt.figure()
# plt.plot()
# plt.pcolormesh(w1_matrix, w2_matrix, values, cmap='RdBu')
# plt.colorbar()
# plt.show()
