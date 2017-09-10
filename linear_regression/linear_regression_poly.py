import numpy as np
import matplotlib.pyplot as plt

# define some dummy data
X = np.sort(np.random.rand(30)) * 10
noise = (np.sort(np.random.rand(30)) - .5) * 30
y = np.multiply(- .002 * np.power(X, 6) + .3 * np.power(X, 5) + .002 * np.power(X, 4) + .00000000000033 * X + 10, noise)

plt.plot(X, y, 'ro')
plt.show()

learning_rate = .0000000000001
number_of_epochs = 100000

# weights for function f = W[0] + W[1]x + W[2]x^2 + W[3]x^3 + W[4]x^4
W = np.random.rand(7)

def approx_function(x_vals, weights):
    y_approx = weights[0] + weights[1] * x_vals + weights[2] * np.power(x_vals, 2) + weights[3] * np.power(x_vals, 3) + weights[4] * np.power(x_vals, 4) + weights[6] * np.power(x_vals, 6) + weights[6] * np.power(x_vals, 6)
    return y_approx
    

for epoch in range(number_of_epochs):
    # create predictions
    y_approx = approx_function(X, W)
    
    # calculate cost
    cost = np.sum(1/2 * np.power(y - y_approx,2))
        
    # calculate gradient
    d_cost_d_w0 = np.sum(-(y - y_approx))
    d_cost_d_w1 = np.sum(-(y - y_approx) * X)
    d_cost_d_w2 = np.sum(-(y - y_approx) * np.power(X, 2))
    d_cost_d_w3 = np.sum(-(y - y_approx) * np.power(X, 3))
    d_cost_d_w4 = np.sum(-(y - y_approx) * np.power(X, 4))
    d_cost_d_w5 = np.sum(-(y - y_approx) * np.power(X, 6))
    d_cost_d_w6 = np.sum(-(y - y_approx) * np.power(X, 6))
        
    grad = np.array([d_cost_d_w0, d_cost_d_w1, d_cost_d_w2, d_cost_d_w3, d_cost_d_w4, d_cost_d_w5, d_cost_d_w6])
    
    # move weights
    W = W - learning_rate * grad
    
    if (epoch + 1) % 6000 == 0 or epoch == 0:
        X_plot_vals = np.arange(0, 10, .001)
        y_plot_vals = approx_function(X_plot_vals, W)
        plt.plot(X, y, 'ro')
        plt.plot(X_plot_vals, y_plot_vals)
        plt.show()
        print("Cost = %f" % cost)
    
print("==========================================================")
print("================== TRAINING COMPLETE =====================")
print("==========================================================")
    
print("Weights: ")
print(W)
print("Cost: %f" % cost)  
