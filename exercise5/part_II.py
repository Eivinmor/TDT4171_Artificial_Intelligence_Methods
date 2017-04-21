import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import time


def logistic(w, x):
    return 1 / (1 + np.exp(- np.inner(w, x)))


def classify(w, x):
    x = np.hstack(([1], x))
    return 0 if (logistic(w, x) < 0.5) else 1


def batch_train_w(x_train, y_train, learn_rate=0.1, niter=1000):
    start_time = time.time()
    x_train = np.hstack((np.array([1]*x_train.shape[0]).reshape(x_train.shape[0], 1), x_train))
    dim = x_train.shape[1]
    num_n = x_train.shape[0]
    w = np.random.rand(dim)
    for it in range(niter):
        if it % (niter/10) == 0:
            print(100 * it / niter, "\b%")
        for i in range(dim):
            update_grad = 0.0
            for n in range(num_n):
                logi_val = logistic(w, x_train[n])
                # something needs to be done here
                update_grad += (y_train[n] - logi_val) * logi_val * (1 - logi_val) * x_train[n][i]
            w[i] += learn_rate * update_grad / num_n
    print("Batch train execution time: ", time.time() - start_time, "seconds")
    return w


def stochast_train_w(x_train, y_train, learn_rate=0.1, niter=1000):
    start_time = time.time()
    x_train = np.hstack((np.array([1]*x_train.shape[0]).reshape(x_train.shape[0], 1), x_train))
    dim = x_train.shape[1]
    num_n = x_train.shape[0]
    w = np.random.rand(dim)
    index_lst = []
    for it in range(niter):
        if it % (niter/10) == 0:
            print(100 * it / niter, "\b%")
        if len(index_lst) == 0:
            index_lst = random.sample(range(num_n), k=num_n)
        xy_index = index_lst.pop()
        x = x_train[xy_index, :]
        y = y_train[xy_index]
        for i in range(dim):
            logi_val = logistic(w, x)
            # TODO Hvorfor må x[i] være negativ her, men ikke i part_I?
            update_grad = (y - logi_val) * logi_val * (1 - logi_val) * x[i]  # ## something needs to be done here
            w[i] += learn_rate * update_grad  # ## something needs to be done here
    print("Stochastic train execution time: ", time.time() - start_time, "seconds")
    return w


def train_and_plot(xtrain, ytrain, xtest, ytest, training_method, learn_rate=0.1, niter=10):
    # train data
    data = pd.DataFrame(np.hstack((xtrain, ytrain.reshape(xtrain.shape[0], 1))), columns=['x', 'y', 'lab'])
    ax = data.plot(kind='scatter', x='x', y='y', c='lab', cmap=cm.copper, edgecolors='black')

    # train weights
    w = training_method(xtrain, ytrain, learn_rate, niter)
    error = []
    y_est = []
    for i in range(len(ytest)):
        error.append(np.abs(classify(w, xtest[i]) - ytest[i]))
        y_est.append(classify(w, xtest[i]))
    y_est = np.array(y_est)
    data_test = pd.DataFrame(np.hstack((xtest, y_est.reshape(xtest.shape[0], 1))), columns=['x', 'y', 'lab'])
    data_test.plot(kind='scatter', x='x', y='y', c='lab', ax=ax, cmap=cm.coolwarm, edgecolors='black')
    print("error=", np.mean(error))
    plt.show()
    return w


def train_and_test(xtrain, ytrain, xtest, ytest, training_method, learn_rate=0.1, niter=10):
    # train weights
    start_time = time.time()
    w = training_method(xtrain, ytrain, learn_rate, niter)
    run_time = time.time() - start_time
    error = []
    y_est = []
    for i in range(len(ytest)):
        error.append(np.abs(classify(w, xtest[i]) - ytest[i]))
        y_est.append(classify(w, xtest[i]))
    y_est = np.array(y_est)
    print("error=", np.mean(error))
    return run_time, np.mean(error)


def read_file(filename):
    x_list = []
    y_list = []
    file = open("data/" + filename + ".csv")
    for line in file:
        line_list = line.strip("\n").split("\t")
        x_list.append([float(line_list[0]), float(line_list[1])])
        y_list.append(float(line_list[2]))
    x_array = np.array(x_list)
    y_array = np.array(y_list)
    return x_array, y_array

x_train, y_train = read_file("data_big_nonsep_train")
x_test, y_test = read_file("data_big_nonsep_test")
# train_and_plot(x_train, y_train, x_test, y_test, batch_train_w, niter=200)
# train_and_plot(x_train, y_train, x_test, y_test, stochast_train_w, niter=10)

time_list = []
error_list = []
iter_list = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
for i in iter_list:
    print("Running with", i, "iterations.")
    run_time, error = train_and_test(x_train, y_train, x_test, y_test, stochast_train_w, niter=i)
    time_list.append(run_time)
    error_list.append(error)

# plt.ylabel('Error')
# plt.plot(iter_list, error_list)

plt.ylabel('Time (s)')
plt.plot(iter_list, time_list)

plt.xlabel('Iterations')
plt.show()
