import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd import grad

csvname = 'kleibers_law_data.csv'
data = np.loadtxt(csvname,delimiter = ',')
fig, ax = plt.subplots(1, 1, figsize=(4,4))
ax.scatter(np.log(data[0,:]),np.log(data[1,:]),color = 'k',edgecolor = 'w')
plt.show()
x = np.log(data[:-1,:])
y = np.log(data[-1,:])
def model(x,w):
    a = w[0] +np.dot(x.T,w[1:])
    return a.T
def least_squares(w):
    cost = np.sum((model(x,w)-y)**2)
    return cost/float(y.size)
w = np.asarray([1.5,1.5])
w1=least_squares(w)
print(w1)
def gradient_descent(g,alpha,max_its,w):
     gradient = grad(g)
     weight_history=[]
     best_w = w
     best_eval = g(w)
     for k in range(max_its):
         grad_eval = gradient(w)
         weight_history.append(w)
         w = w - alpha*grad_eval
         test_eval = g(w)
         if test_eval < best_eval:
             best_eval = test_eval
             best_w = w
     return best_w,weight_history
best_w1,weight_history1 = gradient_descent(g = least_squares,alpha = 10**-0,max_its = 500,w = w)
best_w2,weight_history2 = gradient_descent(g = least_squares,alpha = 10**-1,max_its = 500,w = w)
best_w3,weight_history3 = gradient_descent(g = least_squares,alpha = 10**-2,max_its = 500,w = w)
def MSE(weight_history,g):
 # loop over weight history and compute the MSE at each step o gradient descent
 costfunchistory=[g(w) for w in weight_history]
 # plot cost function history
 plt.figure()
 plt.plot([i for i in range(500)], costfunchistory)
 plt.show()
MSE(weight_history1, g = least_squares)
MSE(weight_history2, g = least_squares)
MSE(weight_history3, g = least_squares)
w=best_w3
# scatter plot the input data
fig, ax = plt.subplots(1, 1, figsize=(4,4))
ax.scatter(np.log(data[0,:]),np.log(data[1,:]),color = 'k',edgecolor = 'w')
# fit a trend line
x_vals = np.linspace(-7.5,7.5,200)
y_vals = w[0] + w[1]*x_vals
ax.plot(x_vals,y_vals,color = 'r')
plt.show()
print("w0,w1")
print(w[0], w[1])

# write the equation and plot it
A = np.vstack([x, np.ones(len(x[0]))]).T
m, c = np.linalg.lstsq(A, y, rcond=None)[0]
_ = plt.plot(x[0], y, 'o', label='Original data', markersize=10)
_ = plt.plot(x[0], m*x[0] + c, 'r', label='Fitted line')
_ = plt.legend()


# predict
# y1=m*10+c
y_p = w[0] + w[1]*10
calories = y_p*1000/4.18
# calories = y1*1000/4.18
print("calories")
print(calories)

plt.show()