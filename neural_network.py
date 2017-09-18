import numpy as np
import matplotlib.pyplot as plt
from training_model import TrainingModel


class NeuralNetwork(TrainingModel):
    '''
    basic neural network
    '''

    def __init__(self, layers, X, y, learning_rate = 1, weight_decay = 0, activation_fn=TrainingModel.sigmoid, number_of_epochs=10, plot_cost_graph=False, number_of_batches=10):
        '''
        inputs:

        layers            - A list containing the number of nodes in each
                            layer.

        X                 - inputs for the neural network.  This is a numpy
                            array of shape (#examples X #features).  Note:
                            #features must be equal to the value first index of
                            layers

        y                 - outputs for the neural network.  This is a numpy
                            array of shape (#examples X #outputs).  Note 1:
                            #outputs must be equal to the value last index of
                            layers.  Note 2: the number of rows in X must match
                            the number of rows in y.

        learning_rate     - A floating point number.  How aggressively the
                            weights of the neural network should be updated.

        weight_decay      - A floating point number.  Regularization term

        activation_fn     - One of the functions defined in TrainingModel.
                            Options are:

                                * sigmoid
                                * tanh
                                * rect_lin

        number_of_epochs  - Interger.  Number of iterations to run

        plot_cost_graph   - A boolean deciding if the cost function should be
                            plotted, good for debugging.

        number_of_batches - Interger.  The number of batches to train with.  The
                            larger number the slower the model will train; the
                            smaller the number, the faster the number of, but
                            the more ram is used.  This should be between 1 and
                            the number of examples.
        '''

        self.plot_cost_graph = plot_cost_graph

        self.layers = layers

        self.number_of_epochs = number_of_epochs

        self.learning_rate = learning_rate

        if X.shape[0] != y.shape[0]:
            print("number of examples in input does not match number of examples in output!")
            return

        self.weight_decay = weight_decay

        self.X = X

        self.y = y

        # weights is a list of matricies.  each matrix is of shape (layer X next layer)
        self.weights = []

        # biases is an array of shape (1 X next_layer)
        self.biases = []

        # calculate weights, place these in a list of np arrays
        for layer, next_layer in zip(layers, layers[1:]):
            self.weights.append(np.random.randn(layer, next_layer)/100)

            # bias value for everything except the output layer
            self.biases.append(np.random.randn(next_layer).reshape(1, next_layer))

        # gradient
        self.grad = []

        # map of functions and their deratives
        derivative_dict = {self.tanh: self.tanh_prime, self.sigmoid: self.sigmoid_prime, self.rect_lin: self.rect_lin_prime}

        self.activation_fn = activation_fn
        self.activation_fn_prime = derivative_dict[activation_fn]

        self.number_of_batches = number_of_batches
        self.batch_size = int(np.round(self.X.shape[0]/self.number_of_batches))


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

        # TODO: the cost is currently being calculated assuming the logistic
        # function is being used of activation.  This should be more flexible

        # number of examples
        m = self.X.shape[0]

        reg_term = 0
        for weight in self.weights:
            reg_term += np.sum(np.power(weight, 2))

        _, _, y_approx = self.foward_feed(self.X)
        #cost = 1/m * np.sum(1/2 * np.power(y_approx - self.y, 2)) + self.weight_decay / 2 * reg_term
        cost = 1/m * np.sum(- self.y * np.log(y_approx) - (1 - self.y) * np.log(1 - y_approx))
        return cost


    def back_prop(self, x, y):
        '''
        perform back propigation to updated the weights

        x - numpy array of shape (batch size X #number of features) a batch of
            inputs to train with

        y - numpy array of shape (batch size X #number of outputs) the batch of
            corresponding outputs to train with
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
        delta[n] = -(y - a[n])# * self.activation_fn_prime(z[n])

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




