import numpy as np
import matplotlib.pyplot as plt
from training_model import TrainingModel


class NeuralNetwork(TrainingModel):
    '''
    basic neural network
    '''

    def __init__(self, layers, X, y, learning_rate = 1, weight_decay = 0, activation_fn=TrainingModel.sigmoid, number_of_epochs=10, plot_cost_graph=False, number_of_batches=10):


        self.plot_cost_graph = plot_cost_graph

        # layers should be a list (or np array) of sizes, where each index
        # represents a layer from left to right, and the number is the number of
        # nodes
        self.layers = layers
        self.number_of_epochs = number_of_epochs
        # the rate at which we want to perform gradient descent
        self.learning_rate = learning_rate

        # X and y should have the same number of examples!
        assert X.shape[0] == y.shape[0]

        # regularization parameter.  real number
        self.weight_decay = weight_decay

        # X is expected to be a matrix of shape (examlpes X features)
        self.X = X

        # y is expected to be an array of shape (examples X 1)
        self.y = y

        # weights is a list of matricies.  each matrix is of shape (layer X next layer)
        self.weights = []

        # biases is an array of shape (1 X next_layer)
        self.biases = []

        # calculate weights, place these in a list of np arrays
        for layer, next_layer in zip(layers, layers[1:]):
            self.weights.append(np.random.rand(layer, next_layer))

            # bias value for everything except the output layer
            self.biases.append(np.random.rand(next_layer).reshape(1, next_layer))

        # gradient
        self.grad = []

        # map of functions and their deratives
        derivative_dict = {self.tanh: self.tanh_prime, self.sigmoid: self.sigmoid_prime, self.rect_lin: self.rect_lin_prime}

        self.activation_fn = activation_fn
        self.activation_fn_prime = derivative_dict[activation_fn]

        self.number_of_batches = number_of_batches
        self.batch_size = int(np.round(self.X.shape[0]/self.number_of_batches))

        #super(NeuralNetwork, self).__init__()


    def foward_feed(self, x):
        '''
        given starting values in X, calculate the values of each neuron in the
        neural network in each layer.

        returns a list of the neuron values by layer
        '''

        # a is the input values of the current layer.  We will start with X
        layer_activations = []

        layer_output = []

        a = x
        layer_output.append(a)

        for weight, bias in zip(self.weights, self.biases):
            z = np.matmul(a, weight) + bias
            a = self.activation_fn(z)

            layer_activations.append(z)
            layer_output.append(a)

        y_approx = a

        return layer_activations, layer_output, y_approx


    def cost(self):
        '''
        compute to cost
        '''

        # number of examples
        m = self.X.shape[0]

        reg_term = 0
        for weight in self.weights:
            reg_term += np.sum(np.power(weight, 2))

        _, _, y_approx = self.foward_feed(self.X)
        cost = 1/m * np.sum(1/2 * np.power(y_approx - self.y, 2)) + self.weight_decay / 2 * reg_term
        #cost = 1/m * np.sum(- self.y * np.log(y_approx) - (1 - self.y) * np.log(1 - y_approx))
        return cost


    def back_prop(self, x, y):
        '''
        perform back propigation to updated the weights
        '''

        # the list of deltas
        delta = [None] * len(self.layers)
        grad = [None] * (len(self.layers) - 1)
        bias_grad = [None] * (len(self.layers) - 1)

        z, a, _ = self.foward_feed(x)

        # pad z's to match number of layers
        z.insert(0, None)

        # shorthand for clarity
        W = self.weights

        # number of layers - 1 since we are indexing starting at 0
        n = len(self.layers) - 1

        # calculate the gradient of the output layer
        delta[n] = -(y - a[n]) * self.activation_fn_prime(z[n])

        for l in range(n - 1, 0, -1):
            delta[l] = np.matmul(delta[l + 1] , W[l].T) * self.activation_fn_prime(z[l])

        for l in range(n - 1, -1, -1):
            grad[l] = np.matmul(a[l].T, delta[l + 1])
            bias_grad[l] = np.sum(delta[l + 1], axis=0)

        self.grad = grad
        self.bias_grad = bias_grad


    def train_model(self):
        no_exs = self.X.shape[0]

        for epoch in range(self.number_of_epochs):
            start_of_batch = 0
            for i in range(self.number_of_batches):

                end_of_batch = min(start_of_batch + self.batch_size, self.X.shape[0] - 1)

                self.back_prop(self.X[start_of_batch:end_of_batch,:], self.y[start_of_batch:end_of_batch,:])
                start_of_batch = start_of_batch + self.batch_size + 1

                for j, _ in enumerate(self.weights):
                    self.weights[j] = self.weights[j] - self.learning_rate * (1/no_exs * self.grad[j] + self.weight_decay * self.weights[j])
                    self.biases[j] = self.biases[j] - self.learning_rate * (1/no_exs * self.bias_grad[j])

            if self.plot_cost_graph:
                if epoch % np.ceil(self.number_of_epochs/100) == 0:
                    plt.plot(epoch, self.cost(), 'ro')

            if epoch % np.ceil(self.number_of_epochs/10) == 0:
                print(self.cost())


        if self.plot_cost_graph:
            plt.show()




