import autograd.numpy as np
from autograd.misc.flatten import flatten_func
from autograd import grad as compute_grad
import matplotlib.pyplot as plt
import copy

# load data
csvname = 'credit_dataset.csv'
data = np.loadtxt(csvname, delimiter=',')
x = data[0:20, :]
y = data[20, :]
x = x.T


# gradient descent
def gradient_descent(g, w, alpha, max_its, beta, version):
    g_flat, unflatten, w = flatten_func(g, w)
    grad = compute_grad(g_flat)

    w_hist = []
    w_hist.append(unflatten(w))

    z = np.zeros((np.shape(w)))

    for k in range(max_its):
        grad_eval = grad(w)
        grad_eval.shape = np.shape(w)

        if version == 'normalized':
            grad_norm = np.linalg.norm(grad_eval)
            if grad_norm == 0:
                grad_norm += 10 ** -6 * np.sign(2 * np.random.rand(1) - 1)
            grad_eval /= grad_norm

        z = beta * z + grad_eval
        w = w - alpha * z

        w_hist.append(unflatten(w))
    return w_hist


# normalize the data
x_means = np.mean(x, axis=0)
x_stds = np.std(x, axis=0)


def normalize(data, data_mean, data_std):
    normalized_data = (data - data_mean) / data_std
    return normalized_data


x_orig = copy.deepcopy(x)
x_norm = normalize(x, x_means, x_stds)

# add ones to the data
one = np.ones(len(x))
one = one.reshape((-1, 1))
x_orig = np.hstack((one, x))
x_norm = np.hstack((one, x_norm))
print(x_norm)
print(x_norm.shape[0])
print(x_norm.shape[1])


# Perceptron cost function
def perceptron(w):
    cost = np.sum(np.maximum(0, -y * np.dot(x_norm, w)))
    return cost / float(np.size(y))


# print the weight history
w = np.ones((21))
alpha = 0.1
max_its = 100
weight_history = gradient_descent(perceptron, w, alpha, max_its, beta=0, version='normalized')
weight_history = np.delete(weight_history, (0), axis=0)
print(weight_history)
print(weight_history.shape[0])
print(weight_history.shape[1])
print(weight_history[1])
print(type(weight_history[1]))




# calculate the number of misclassification
def number(y, weight_history):
    table=np.zeros((2,2),dtype=np.int)
    numbers = np.zeros(len(weight_history))
    for i in range(len(weight_history)):
        y_pre = np.dot(x_norm, weight_history[i])
        number = 0
        for j in range(len(y)):
            if np.sign(y[j]) == np.sign(y_pre[j]):
                number += 0
            else:
                number += 1
        numbers[i] = number
    # zhaodao zuixiao
    p=np.argmin(numbers)
    # # fenlei
    # tz=0
    # tf=0
    # f_tz=0
    # f_tf=0
    # y_pret = np.dot(x_norm, weight_history[p])
    # for j in range(len(y)):
    #     if y[j]>0:
    #         if np.sign(y[j]) == np.sign(y_pret[j]):
    #             tz += 1
    #         else:
    #             f_tz += 1
    #     else:
    #         if np.sign(y[j]) == np.sign(y_pret[j]):
    #             tf += 1
    #         else:
    #             f_tf += 1
    # table[0][0]=tz
    # table[0][1]=f_tz
    # table[1][0]=f_tf
    # table[1][1]=tf
    #
    # print('the table is: ',table)
    return numbers


# plot figure and print the minimum number of classification
fig, ax = plt.subplots(1, 1, figsize=(6, 3))
ax.plot(np.linspace(0, 100, 100), number(y, weight_history), 'b')
plt.xlabel('iteration')
plt.ylabel('number of misclassifications')
plt.show()
print('The minimum number of misclassifications is', np.amin(number(y, weight_history)))