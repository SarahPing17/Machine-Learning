# This file is associated with the book
# "Machine Learning Refined", Cambridge University Press, 2016.
# by Jeremy Watt, Reza Borhani, and Aggelos Katsaggelos.

import numpy as np
import matplotlib.pyplot as plt
import csv


# sigmoid for softmax/logistic regression minimization
def sigmoid(z):
    y = 1 / (1 + np.exp(-z))
    return y


# import training data
def load_data(csvname):
    # load in data
    reader = csv.reader(open(csvname, "r"), delimiter=",")
    d = list(reader)

    # import data and reshape appropriately
    data = np.array(d).astype("float")
    X = data[:, 0:2]
    y = data[:, 2]
    y.shape = (len(y), 1)

    # pad data with ones for more compact gradient computation
    o = np.ones((np.shape(X)[0], 1))
    X = np.concatenate((o, X), axis=1)
    X = X.T

    return X, y


# YOUR CODE GOES HERE - create a newton method for softmax cost/logistic regression
def softmax_newton(X, y):
    w = np.random.randn(3, 1)
    iter = 1
    max_its = 5
    grad = 1
    beta1 = np.array([[1], [1], [1], [1], [1]])
    beta2 = np.ones((50, 1))
    beta = np.concatenate((beta1, beta2), axis=0)
    while np.linalg.norm(grad) > 10 ** (-12) and iter < max_its:
        # calculate newton method
        r_0 = np.dot(X.T, w)
        r_1 = -y*r_0
        r_2 = -sigmoid(r_1)
        r_3 = r_2*y
        r = r_3*beta
        grad_soft = np.dot(X, r)
        t_0 = np.dot(X.T,w)
        t_1 = -y*t_0
        t_2 = sigmoid(t_1)
        t_3 = 1 - t_2
        t_4 = t_2 * t_3 *beta
        t_5 = t_4 * (X.T)
        grad2_soft = np.dot(t_5.T, X.T)
        w = w - np.linalg.pinv(grad2_soft).dot(grad_soft)
        iter += 1

    return w


# plots everything
def plot_all(X, y, w):
    # custom colors for plotting points
    red = [1, 0, 0.4]
    blue = [0, 0.4, 1]

    # scatter plot points
    fig = plt.figure(figsize=(4, 4))
    ind = np.argwhere(y == 1)
    ind = [s[0] for s in ind]
    plt.scatter(X[1, ind], X[2, ind], color=red, edgecolor='k', s=25)
    ind = np.argwhere(y == -1)
    ind = [s[0] for s in ind]
    plt.scatter(X[1, ind], X[2, ind], color=blue, edgecolor='k', s=25)
    plt.grid('off')

    # plot separator
    s = np.linspace(0, 1, 100)
    plt.plot(s, (-w[0] - w[1] * s) / w[2], color='k', linewidth=2)

    # clean up plot
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.show()


# load in data
X, y = load_data('3d_classification_data_v2_mbalanced2.csv')

# run newton method
w = softmax_newton(X, y)
print(X)
print(y)
print(w)
# plot points and separator
plot_all(X, y, w)