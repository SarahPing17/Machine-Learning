# import numpy as np
import matplotlib.pyplot as plt
import csv
import autograd.numpy as np
from autograd import grad
from autograd import hessian
import autograd.numpy as np
from autograd.misc.flatten import flatten_func
from autograd import grad as compute_grad
import matplotlib.pyplot as plt
import copy


# sigmoid for softmax/logistic regression minimization
def sigmoid(z):
    y = 1 / (1 + np.exp(-z))
    return y


def model(x, w):
    a = w[0] + np.dot(x.T, w[1:])
    return a.T


def softmax(x, w):
    cost = np.sum(np.log(1 + np.exp(-y * model(x, w))))
    return cost / float(np.size(y))


# import training data
def load_data(csvname):
    # load in data
    reader = csv.reader(open(csvname, "r"), delimiter=",")
    d = list(reader)

    # import data and reshape appropriately
    data = np.array(d).astype("float")
    X = data[:, 0:8]
    y = data[:, 8]
    y.shape = (len(y), 1)

    # pad data with ones for more compact gradient computation
    o = np.ones((np.shape(X)[0], 1))
    X = np.concatenate((o, X), axis=1)
    X = X.T

    return X, y


# load in data
X, y = load_data('breast_cancer_data.csv')


# create a gradient descent function for softmax
def softmax_gradient(X, y):
    # define initial w for softmax and squared margin
    w_soft = (np.random.randn(9, 1)) / 35

    # define some common parameters
    X = X / 35
    y = y
    max_its = 1000

    # define parameters of softmax
    grad_soft = 1
    ite_soft = 0
    it_soft = []
    misc_soft = []

    alpha = 0.1
    # softmax iteration
    while np.linalg.norm(grad_soft) > 10 ** (-12) and ite_soft < max_its:
        # while ite_soft < max_its:
        # calculate gradient
        r_0 = np.dot(X.T, w_soft)
        r_1 = -y * r_0
        r_2 = -sigmoid(r_1)
        r_3 = r_2 * y
        r = r_3
        grad_soft = np.dot(X, r)

        # calculate misclassfication
        t_0 = np.dot(X.T, w_soft)
        t_1 = -y * t_0
        misclass_soft = np.sign(t_1)
        misclass_soft_new = (misclass_soft > 0).sum()
        misc_soft.append(misclass_soft_new)

        # calculate new w
        w_soft = w_soft - alpha * grad_soft

        # calculate iteration
        ite_soft += 1
        it_soft.append(ite_soft)

    return it_soft, misc_soft

#
# def gradient_descent(g, w, alpha, max_its, beta, version):
#     grad = compute_grad(g)
#
#     w_hist = []
#     w_hist.append(w)
#
#     z = np.zeros((np.shape(w)))
#
#     for k in range(max_its):
#         grad_eval = grad(w)
#         grad_eval.shape = np.shape(w)
#
#         if version == 'normalized':
#             grad_norm = np.linalg.norm(grad_eval)
#             if grad_norm == 0:
#                 grad_norm += 10 ** -6 * np.sign(2 * np.random.rand(1) - 1)
#             grad_eval /= grad_norm
#
#         z = beta * z + grad_eval
#         w = w - alpha * z
#
#         w_hist.append(unflatten(w))
#     return w_hist
#
#
# # Perceptron cost function
# def perceptron(w):
#     cost = np.sum(np.maximum(0, -y * np.dot(X, w)))
#     return cost / float(np.size(y))
#
#
# # print the weight history
# w = np.ones((9))
# # w0 = (np.random.randn(9, 1)) / 35
# alpha = 0.1
# max_its = 100
# weight_history = gradient_descent(perceptron, w, alpha, max_its, beta=0, version='normalized')
# weight_history = np.delete(weight_history, (0), axis=0)
#
#
#
# # calculate the number of misclassification
# def number(y, weight_history):
#     numbers = np.zeros(len(weight_history))
#     for i in range(len(weight_history)):
#         y_pre = np.dot(x_norm, weight_history[i])
#         number = 0
#         for j in range(len(y)):
#             if np.sign(y[j]) == np.sign(y_pre[j]):
#                 number += 0
#             else:
#                 number += 1
#         numbers[i] = number
#     return numbers


# plots everything
def plot_all(a_ite, a_misclass):
    plt.plot(a_ite[1:], a_misclass[1:])
    plt.xlabel('iteration')
    plt.ylabel('number of misclassification of softmax')
    plt.legend([r'$softmax\:cost$', r'$squared\:margin\:perceptron$'])
    plt.show()


# get the iteration and misclassification of softmax
a = softmax_gradient(X, y)
a_ite = a[0]
a_misclass = a[1]

# get the iteration and misclassification of squared margin
# b = softmax_gradient(X, y)
# b_ite = b[2]
# b_misclass = b[3]

# plot points and separator
plot_all(a_ite, a_misclass)
