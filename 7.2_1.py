import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd import grad

# load data
csvname = '4class_data.csv'
data = np.loadtxt(csvname, delimiter=',')
x = data[:-1, :]
y = data[-1, :]
x = x.T

print('y10',y[37])

y0=np.zeros(len(y))
y1=np.zeros(len(y))
y2=np.zeros(len(y))
y3=np.zeros(len(y))

for k in range(len(y)):
    if y[k] == 0:
        y0[k] = 1
    else:
        y0[k] = -1
y0.shape = (len(y0), 1)
print("y0", y0)

for k in range(len(y)):
    if y[k] == 1:
        y1[k] = 1
    else:
        y1[k] = -1



for k in range(len(y)):
    if y[k] == 2:
        y2[k] = 1
    else:
        y2[k] = -1
y2.shape = (len(y2), 1)

for k in range(len(y)):
    if y[k] == 3:
        y3[k] = 1
    else:
        y3[k] = -1
y3.shape = (len(y3), 1)

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

# sigmoid for softmax/logistic regression minimization
def sigmoid(z):
    y = 1 / (1 + np.exp(-z))
    return y

def softmax_grad(X,y):
    w = np.random.randn(3,1)
    alpha = 10**(-2)
    iter = 1
    max_its = 1000
    grad = 1
    while np.linalg.norm(grad) > 10**(-12) and iter < max_its:
        e = np.dot(X.T,w)
        q = -y*e
        r_1 = -sigmoid(q)
        r_2 = r_1*y
        r = r_2
        grad = np.dot(X,r)
        w = w - alpha*grad
        iter += 1

    return w

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
    return best_w

#
# # normalize the data
# x_means = np.mean(x, axis=0)
# x_stds = np.std(x, axis=0)
#
#
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
print("x_orig.T",x_orig.T)
# print(x_norm.shape[0])
# print(x_norm.shape[1])


# # Perceptron cost function
# def perceptron(w):
#     cost = np.sum(np.maximum(0, -y * np.dot(x_norm, w)))
#     return cost / float(np.size(y))

# Perceptron cost function
def perceptron0(w):
    cost = np.sum(np.maximum(0, -y0 * np.dot(x_orig, w)))
    return cost / float(np.size(y0))
def perceptron1(w):
    cost = np.sum(np.maximum(0, -y1 * np.dot(x_orig, w)))
    return cost / float(np.size(y1))
def perceptron2(w):
    cost = np.sum(np.maximum(0, -y2 * np.dot(x_orig, w)))
    return cost / float(np.size(y2))
def perceptron3(w):
    cost = np.sum(np.maximum(0, -y3 * np.dot(x_orig, w)))
    return cost / float(np.size(y3))


# print the weight history
w = np.ones((3))
alpha = 0.1
max_its = 100
# weight_history = gradient_descent(perceptron, w, alpha, max_its, beta=0, version='normalized')
# weight_history = np.delete(weight_history, (0), axis=0)
weight0 = softmax_grad(x_orig.T, y0)
weight1 = gradient_descent(perceptron1, alpha, max_its, w)
weight1.shape = (len(weight1), 1)
weight2 = softmax_grad(x_orig.T, y2)
weight3 = softmax_grad(x_orig.T, y3)
print("weight0",weight0)
print("weight1",weight1)
# print(weight_history.shape[0])
# print(weight_history.shape[1])
# print(weight_history[1])
# print(type(weight_history[1]))
# for i in range(len(weight0)):
#     y0123=np.array([np.dot(x_orig,weight0),np.dot(x_orig,weight1),np.dot(x_orig,weight2),np.dot(x_orig,weight3)])
# yall=np.argmax(y0123)
# print("yall",yall)

print('y',y)
number = 0
# calculate the number of misclassification
    # numbers = np.zeros(len(x_orig))
for i in range(len(y)):
        # index= np.zeros(len(y))
        # y0123 = np.zeros(4,len(y))
        y0123 = np.array([np.dot(x_orig[i], weight0/np.linalg.norm(weight0)),
                             np.dot(x_orig[i], weight1/np.linalg.norm(weight1)),
                             np.dot(x_orig[i], weight2/np.linalg.norm(weight2)),
                             np.dot(x_orig[i], weight3/np.linalg.norm(weight3))])
        print(y0123)
        # y0123.shape = (4,1)
        index = np.argmax(y0123,axis=0)
        print(index[0])
        print(type(index[0]))
        print(y[i])
        y[i].astype(int)
        print(type(y[i]))

        if index[0] == y[i]:
            number += 0
        if index[0]!=y[i]:
            number += 1
    # zhaodao zuixiao
    # p = np.argmin(numbers)
print("number",number)
print("index",index)



# plot figure and print the minimum number of classification
# fig, ax = plt.subplots(1, 1, figsize=(6, 3))
# ax.plot(np.linspace(0, 100, 100), number(y, weight_history), 'b')
# plt.xlabel('iteration')
# plt.ylabel('number of misclassifications')
# plt.show()
# print('The minimum number of misclassifications is', np.amin(number(y, weight_history)))


# plots everything
def plot_all(X, y, w0,w1,w2,w3):
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
    ind = np.argwhere(y == 3)
    ind = [s[0] for s in ind]
    plt.scatter(X[ind,1], X[ind,2], color='yellow', edgecolor='k', s=25)
    plt.grid('off')

    # plot separator
    s = np.linspace(0, 1, 100)
    plt.plot(s, (-w0[0] - w0[1] * s) / w0[2], color='k', linewidth=2)
    plt.plot(s, (-w1[0] - w1[1] * s) / w1[2], color='k', linewidth=2)
    plt.plot(s, (-w2[0] - w2[1] * s) / w2[2], color='k', linewidth=2)
    plt.plot(s, (-w3[0] - w3[1] * s) / w3[2], color='k', linewidth=2)

    # clean up plot
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.show()

plot_all(x_orig, y, weight0, weight1, weight2, weight3)