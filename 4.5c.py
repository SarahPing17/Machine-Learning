# using an automatic differentiator - like the one imported via the statement below - makes coding up gradient descent a breeze
import autograd.numpy as np
from autograd import grad
from autograd import hessian
from matplotlib import rcParams

rcParams['figure.autolayout'] = True
import matplotlib.pyplot as plt
import math
import sympy as sy


# newtons method function - inputs: g (input function), max_its (maximum number of iterations), w (initialization)
def newtons_method(g, max_its, w, **kwargs):
    # compute gradient module using autograd
    gradient = grad(g)
    hess = hessian(g)

    # set numericxal stability parameter / regularization parameter
    epsilon = 10 ** (-7)
    if 'epsilon' in kwargs:
        beta = kwargs['epsilon']

    # run the newtons method loop
    weight_history = [w]  # container for weight history
    cost_history = [g(w)]  # container for corresponding cost function history
    for k in range(max_its):
        # evaluate the gradient and hessian
        grad_eval = gradient(w)
        hess_eval = hess(w)

        # reshape hessian to square matrix for numpy linalg functionality
        hess_eval.shape = (int((np.size(hess_eval)) ** (0.5)), int((np.size(hess_eval)) ** (0.5)))

        # solve second order system system for weight update
        A = hess_eval + epsilon * np.eye(w.size)
        b = grad_eval
        w = np.linalg.solve(A, np.dot(A, w) - b)

        # record weight and cost
        weight_history.append(w)
        cost_history.append(g(w))
    return weight_history, cost_history


w1 = np.array([[1], [1]])


def g(w):
    x = np.dot(np.transpose(w), w)
    return np.log(1 + np.exp(x))


max1 = 10
newtons_method(g, max1, w1)
weights, costliest = newtons_method(g, max1, w1)
plt.plot(costliest)
plt.xlabel("interation")
plt.ylabel("w")
plt.show()
