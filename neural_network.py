import numpy as np
import matplotlib.pyplot as plt
from training_model import TrainingModel

DEBUG_GRAD = 0

class NeuralNetwork(TrainingModel):
    '''
    basic foward feed neural network
    '''

    def __init__(self, layers, X, y, learning_rate = 1, weight_decay = 0, activation_fn=TrainingModel.sigmoid, number_of_epochs=10, plot_cost_graph=False, batch_size=100):
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

        # validate inputs
        if X.shape[0] != y.shape[0]:
            print("number of examples in input does not match number of examples in output!")
            return

        self.plot_cost_graph = plot_cost_graph

        self.layers = layers

        self.number_of_epochs = number_of_epochs

        self.learning_rate = learning_rate

        self.weight_decay = weight_decay

        self.X = X

        self.y = y

        # weights is a list of matricies.  each matrix is of shape (layer X next layer)
        self.weights = []

        # biases is an array of shape (1 X next_layer)
        self.biases = []

        # calculate weights, place these in a list of np arrays
        for layer, next_layer in zip(layers, layers[1:]):
            self.weights.append(np.random.randn(layer, next_layer))

            # bias value for everything except the output layer
            self.biases.append(np.random.randn(next_layer).reshape(1, next_layer))

        # gradient
        self.grad = []

        # map of functions and their deratives
        derivative_dict = {self.tanh: self.tanh_prime, self.sigmoid: self.sigmoid_prime, self.rect_lin: self.rect_lin_prime}

        self.activation_fn = activation_fn
        self.activation_fn_prime = derivative_dict[activation_fn]

        self.batch_size = batch_size
        self.number_of_batches = int(np.round(self.X.shape[0]/self.batch_size))


    @staticmethod
    def foward_feed(x, weights, biases, activation_fn):
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

        for weight, bias in zip(weights, biases):
            z = np.matmul(a, weight) + bias
            a = activation_fn(z)

            layer_activations.append(z)
            layer_output.append(a)

        y_approx = a

        # update last layer
        layer_output[-1] = y_approx

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

        _, _, y_approx = self.foward_feed(self.X, self.weights, self.biases, self.activation_fn)

        # quadratic cost function
        #cost = 1/m * np.sum(1/2 * np.power(y_approx - self.y, 2)) + self.weight_decay / 2 * reg_term

        # cross entropy cost function
        cost = - 1/m * np.sum(self.y * np.log(y_approx) + (1 - self.y) * np.log(1 - y_approx)) + self.weight_decay/(2*m) * reg_term
        return cost


    def grad_check(self):
        '''
        calculate the gradient using divided differences.  This is to make sure
        the gradient we are using is sane
        '''
        m = self.X.shape[0]

        epislon = .0001

        grads_approx = []

        low_weights = [np.copy(w) for w in self.weights]
        high_weights = [np.copy(w) for w in self.weights]

        for k, weight in enumerate(self.weights):

            grad_approx = np.zeros(weight.shape)

            for i in range(weight.shape[0]):
                for j in range(weight.shape[1]):
                    low_weights[k][i, j] -= epislon
                    high_weights[k][i, j] += epislon

                    low_reg_term = 0
                    for low_weight in low_weights:
                        low_reg_term += np.sum(np.power(low_weight, 2))
                    _, _, low_approx = self.foward_feed(self.X, low_weights, self.biases, self.activation_fn)

                    high_reg_term = 0
                    for high_weight in high_weights:
                        high_reg_term += np.sum(np.power(high_weight, 2))
                    _, _, high_approx = self.foward_feed(self.X, high_weights, self.biases, self.activation_fn)

                    # TODO: add weight decay into these calculations
                    low_cost = - np.sum(self.y * np.log(low_approx) + (1 - self.y) * np.log(1 - low_approx)) #+ self.weight_decay / 2 * low_reg_term
                    high_cost = - np.sum(self.y * np.log(high_approx) + (1 - self.y) * np.log(1 - high_approx)) #+ self.weight_decay / 2 * high_reg_term

                    grad_approx[i, j] = (high_cost - low_cost) / (2 * epislon)

                    # re adjust values for next iteration
                    low_weights[k][i, j] += epislon
                    high_weights[k][i, j] -= epislon

            grads_approx.append(grad_approx)

        return grads_approx


    def back_prop(self, x, y):
        '''
        perform back propigation to updated the weights

        x - numpy array of shape (batch size X #number of features) a batch of
            inputs to train with

        y - numpy array of shape (batch size X #number of outputs) the batch of
            corresponding outputs to train with
        '''

        # the list of deltas.  There is one for every layer
        delta     = [None] * len(self.layers)

        # the list of gradients.  There is one for every weight
        grad      = [None] * (len(self.layers) - 1)

        # the list of the bias gradients.  There is one for every bias
        bias_grad = [None] * (len(self.layers) - 1)

        # get the set of activations a and outputs z.
        #
        # the first a = X.  the subsequent a's are computed.  There will be an a
        # for every layer
        #
        # the z's are computed from the a's.  The last a is not used to compute
        # z, however, so there is a z for every layer except for first one (the
        # first z belongs to the second layer).
        z, a, _ = self.foward_feed(x, self.weights, self.biases, self.activation_fn)

        # pad z's to since there is no z for the first layer
        z.insert(0, None)

        # shorthand for clarity
        W = self.weights

        # number of layers - 1 since we are indexing starting at 0
        n = len(self.layers) - 1

        m = x.shape[0]

        # delta to be used when using quadratic error function
        #delta[n] = (a[n] - y) * self.activation_fn_prime(z[n])

        # delta to be used when using cross entropy function
        delta[n] = 1/m * (a[n] - y)

        for l in range(n - 1, 0, -1):
            delta[l] = np.matmul(delta[l + 1] , W[l].T) * self.activation_fn_prime(z[l])

        for l in range(n - 1, -1, -1):
            grad[l] = np.matmul(a[l].T, delta[l + 1])
            bias_grad[l] = np.sum(delta[l + 1], axis=0)

        self.grad = grad
        self.bias_grad = bias_grad


    def train_model(self):
        m = self.X.shape[0]

        for epoch in range(self.number_of_epochs):
            start_of_batch = 0

            if DEBUG_GRAD > 0:
                # verify gradient is correct.  commented out on purpose
                grads_approx = self.grad_check()

            for i in range(self.number_of_batches):

                end_of_batch = min(start_of_batch + self.batch_size, self.X.shape[0])

                self.back_prop(self.X[start_of_batch:end_of_batch,:], self.y[start_of_batch:end_of_batch,:])
                start_of_batch = start_of_batch + self.batch_size

                for j, _ in enumerate(self.weights):

                    if DEBUG_GRAD > 0:
                        print("\nself.grad")
                        print(self.grad[j])

                        print("\ngrads_approx")
                        print(grads_approx[j])

                        print("\ndifference")
                        print(self.grad[j] - grads_approx[j])

                    self.weights[j] = self.weights[j] - self.learning_rate * ( self.grad[j] + self.weight_decay * self.weights[j])
                    self.biases[j] = self.biases[j] - self.learning_rate * ( self.bias_grad[j])

            if self.plot_cost_graph:
                if epoch % np.ceil(self.number_of_epochs/100) == 0:
                    plt.plot(epoch, self.cost(), 'ro')

            if epoch % np.ceil(self.number_of_epochs/10) == 0:
                print(self.cost())


        if self.plot_cost_graph:
            plt.show()




