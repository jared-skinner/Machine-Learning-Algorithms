# implementation of linear regression using numpy
import numpy as np
import matplotlib.pyplot as plt

# create random feature set values between 0 and 10
# sorting so values form a line-ish thing.  It is 
# contrived, but it makes a good example...
X = np.sort(np.random.rand(30)) * 10
y = np.sort(np.random.rand(30)) * 7

# plot our feature set
plt.plot(X, y, 'ro')
plt.show()

# set learning rate (the size of steps we want to take when performing gradient descent).
# I started with a learning rate of 0.1, but that was way too large...
learning_rate = 0.001

# initialize weights
W = np.random.rand(2) * 10

for epoch in range(1000):
    # calculate cost
    y_approx = W[0] * X + W[1]
    cost = np.sum(np.power(1/2*(y - y_approx), 2))
 
    # calculate gradient
    # partial wrt W[0] = -(y - (W[0]*X + W[1])) * X
    # partial wrt W[1] = -(y - (W[0]*X + W[1]))
    d_cost_d_w0 = np.sum(-(y - y_approx)*X)
    d_cost_d_w1 = np.sum(-(y - y_approx))
    grad = np.array([d_cost_d_w0, d_cost_d_w1])
    W = W - learning_rate * grad

    if (epoch + 1) % 250 == 0 or epoch == 0:
        print("epoch: %d" % (epoch + 1), "\n cost: %f" % cost)
        plt.plot(X, y, 'ro')
        plt.plot(X, y_approx)
        plt.show()
