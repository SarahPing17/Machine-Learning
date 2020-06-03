import autograd.numpy as np
from autograd.misc.flatten import flatten
from autograd import grad as compute_grad
import matplotlib.pyplot as plt
import copy
import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd import grad

# load data
csvname = '3class_data.csv'
data = np.loadtxt(csvname, delimiter=',')
x = data[:-1, :]
y = data[-1, :]
x = x.T


# # gradient descent
# def gradient_descent(g, w, alpha, max_its, beta, version):
#     g_flat, unflatten, w = flatten_func(g, w)
#     grad = compute_grad(g_flat)
#
#     w_hist = []
#     w_hist.append(unflatten(w))
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


def gradient_descent(g, alpha, max_its, w):
    gradient = grad(g)
    weight_history = []
    best_w = w
    best_eval = g(w)
    for k in range(max_its):
        grad_eval = gradient(w)
        weight_history.append(w)
        w = w - alpha * grad_eval
        test_eval = g(w)
        if test_eval < best_eval:
            best_eval = test_eval
            best_w = w
    return best_w, weight_history

# # normalize the data
# x_means = np.mean(x, axis=0)
# x_stds = np.std(x, axis=0)


# def normalize(data, data_mean, data_std):
#     normalized_data = (data - data_mean) / data_std
#     return normalized_data
#
#
# x_orig = copy.deepcopy(x)
# x_norm = normalize(x, x_means, x_stds)

# add ones to the data
one = np.ones(len(x))
one = one.reshape((-1, 1))
x_orig = np.hstack((one, x))
# x_norm = np.hstack((one, x_norm))
# print(x_norm)
# print(x_norm.shape[0])
# print(x_norm.shape[1])


# # Perceptron cost function
# def perceptron_multi(w):
#     cost = np.sum(np.maximum(0, -y * np.dot(x_norm, w)))
#     return cost / float(np.size(y))

# compute C linear combinations of input point, one per classifier
def model(x,w):
    a = np.dot(x,w)
    return a.T

lam = 10 ** -5  # our regularization paramter


# def multiclass_perceptron(w):
#     # pre-compute predictions on all points
#     all_evals = model(x_orig, w)
#
#     # compute counting cost
#     cost = 0
#     for p in range(len(y)):
#         # pluck out current true label
#         y_p = y[p]
#
#         # update cost summand
#         cost += np.max(all_evals[p, :]) - all_evals[p, int(y_p)]
#
#     # return cost with regularizer added
#     cost += lam * np.linalg.norm(w[1:, :], 'fro') ** 2
#     return cost / float(len(y))

def multiclass_perceptron(w):
    all_evals = model(x_orig,w)
    a = np.max(all_evals,axis=0)
    b=all_evals[y.astype(int).flatten(),np.arange(np.size(y))]
    cost = np.sum(a-b)
    cost = cost + lam*np.linalg.norm(w[1:,:],'fro')**2
    return cost/float(np.size(y))

# print the weight history
# w = np.zeros([3,3])
w = 0.1*np.random.randn(3,3)
# w = 0.1*np.ones([3,3])
alpha = 0.1
max_its = 200
weight_best, weight_history = gradient_descent(multiclass_perceptron, alpha, max_its, w)
weight_history = np.delete(weight_history, (0), axis=0)
# print(type(weight_history))
# print(weight_best)
# print(weight_history)

# print(weight_history.shape[0])
# print(weight_history.shape[1])
# print(weight_history[1])
# print(type(weight_history[1]))




# # calculate the number of misclassification
# def number(y, weight_history):
#     table=np.zeros((2,2),dtype=np.int)
#     numbers = np.zeros(len(weight_history))
#     for i in range(len(weight_history)):
#         y_pre = np.dot(x_orig, weight_history[i])
#         number = 0
#         for j in range(len(y)):
#             if np.sign(y[j]) == np.sign(y_pre[j]):
#                 number += 0
#             else:
#                 number += 1
#         numbers[i] = number
#     # zhaodao zuixiao
#     p=np.argmin(numbers)
#     # # fenlei
#     # tz=0
#     # tf=0
#     # f_tz=0
#     # f_tf=0
#     # y_pret = np.dot(x_norm, weight_history[p])
#     # for j in range(len(y)):
#     #     if y[j]>0:
#     #         if np.sign(y[j]) == np.sign(y_pret[j]):
#     #             tz += 1
#     #         else:
#     #             f_tz += 1
#     #     else:
#     #         if np.sign(y[j]) == np.sign(y_pret[j]):
#     #             tf += 1
#     #         else:
#     #             f_tf += 1
#     # table[0][0]=tz
#     # table[0][1]=f_tz
#     # table[1][0]=f_tf
#     # table[1][1]=tf
#     #
#     # print('the table is: ',table)
#     return numbers
number = 0
# calculate the number of misclassification
#     numbers = np.zeros(len(x_orig))
for i in range(len(y)):
        # index= np.zeros(len(y))
        # y0123 = np.zeros(4,len(y))
        y0123 = np.array([np.dot(x_orig[i], weight_best[:,0]/np.linalg.norm(weight_best[:,0])),
                             np.dot(x_orig[i], weight_best[:,1]/np.linalg.norm(weight_best[:,1])),
                             np.dot(x_orig[i], weight_best[:,2]/np.linalg.norm(weight_best[:,2]))])
        print(y0123)
        # y0123.shape = (4,1)
        index = np.argmax(y0123,axis=0)
        print(index)
        print(type(index))
        print(y[i])
        y[i].astype(int)
        print(type(y[i]))

        if index == y[i]:
            number += 0
        if index!=y[i]:
            number += 1
    # zhaodao zuixiao
    # p = np.argmin(numbers)
print("number",number)
print("index",index)

# # plot figure and print the minimum number of classification
# fig, ax = plt.subplots(1, 1, figsize=(6, 3))
# ax.plot(np.linspace(0, 100, 100), number(y, weight_history), 'b')
# plt.xlabel('iteration')
# plt.ylabel('number of misclassifications')
# plt.show()
# print('The minimum number of misclassifications is', np.amin(number(y, weight_history)))

# plots everything
def plot_all(X, y, w0,w1,w2):
    # custom colors for plotting points
    # red = [1, 0, 0.4]
    # blue = [0, 0.4, 1]
    # green = [0.4, 1, 0]
    # yellow = [1, 0.4, 0]

    # scatter plot points
    fig = plt.figure(figsize=(4, 4))
    ind = np.argwhere(y == 0)
    ind = [s[0] for s in ind]
    plt.scatter(X[ind,1], X[ind,2], color='blue', edgecolor='k', s=25)
    ind = np.argwhere(y == 1)
    ind = [s[0] for s in ind]
    plt.scatter(X[ind,1], X[ind,2], color='red', edgecolor='k', s=25)
    ind = np.argwhere(y == 2)
    ind = [s[0] for s in ind]
    plt.scatter(X[ind,1], X[ind,2], color='green', edgecolor='k', s=25)
    plt.grid('off')

    # plot separator
    s = np.linspace(0, 1, 100)
    plt.plot(s, (-w0[0] - w0[1] * s) / w0[2], color='k', linewidth=2)
    plt.plot(s, (-w1[0] - w1[1] * s) / w1[2], color='k', linewidth=2)
    plt.plot(s, (-w2[0] - w2[1] * s) / w2[2], color='k', linewidth=2)

    # clean up plot
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.show()
print(weight_best[:,0])
# print(weight_best[:,1])
# print(weight_best[:,2])
plot_all(x_orig, y, weight_best[:,0], weight_best[:,1], weight_best[:,2])