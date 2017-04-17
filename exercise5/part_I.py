import numpy as np
import matplotlib.pyplot as plt
import copy


r = np.arange(-6, 6.0001, 0.05)
w1_matrix, w2_matrix = np.meshgrid(r, r)


def logistic(w, x):
    return 1 / (1 + np.exp(- np.inner(w, x)))


def L_simple(w):
    p1 = (logistic(w, [1, 0]) - 1) ** 2
    p2 = (logistic(w, [0, 1])) ** 2
    p3 = (logistic(w, [1, 1]) - 1) ** 2
    return p1 + p2 + p3


def updateWeights(w, learning_rate):
    w[0] += - learning_rate * (L_simple_deriv(w, 0))
    w[1] += - learning_rate * (L_simple_deriv(w, 1))


def L_simple_deriv(w, i):
    a = (logistic(w, [1, 0]) - 1) * (logistic(w, [1, 0])) * (1 - logistic(w, [1, 0])) * [1, 0][i]
    b = (logistic(w, [0, 1]))     * (logistic(w, [0, 1])) * (1 - logistic(w, [0, 1])) * [0, 1][i]
    c = (logistic(w, [1, 1]) - 1) * (logistic(w, [1, 1])) * (1 - logistic(w, [1, 1])) * [1, 1][i]
    return a + b + c


def partI_task1():
    values = [[None for i in range(len(r))] for i in range(len(r))]
    for i in range(len(r)):
        for j in range(len(r)):
            w = [w1_matrix[i][j], w2_matrix[i][j]]
            values[i][j] = L_simple(w)
    fig = plt.figure()
    plt.plot()
    plt.pcolormesh(w1_matrix, w2_matrix, values, cmap='coolwarm')
    plt.colorbar()
    plt.show()


def gradient_descent(learning_rate, iterations, init_w):
    w = copy.copy(init_w)
    L_simple_storage = []
    for i in range(iterations):
        L_simple_storage.append(L_simple(w))
        updateWeights(w, learning_rate)
        # if i % (iterations/20) == 0:
            # print("|", end="", flush=True)
    # print()
    print("Final weights:", "{0:.2f}".format(w[0]), "{0:.2f}".format(w[1]))
    print("Final L_simple value:", "{0:.5f}".format(L_simple_storage[-1]))
    return L_simple_storage


def partI_task3():
    learning_rates = [0.0001, 0.01, 0.1, 1, 10, 100]
    iterations = 10000
    # init_w = [r[random.randint(0, r.size - 1)], r[random.randint(0, r.size - 1)]]
    init_w = [0, 6]

    print("\nIterations:", iterations)
    print("Initial weights:", init_w)

    for lr in learning_rates:
        print("\nRunning GD with lr =", lr)
        values = gradient_descent(lr, iterations, init_w)
        plt.plot(values, label=lr)
    plt.legend(loc=1)
    plt.show()


# partI_task1()
partI_task3()
