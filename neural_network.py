import numpy as np
import matplotlib.pyplot as plt
from training_model import TrainingModel


class NeuralNetwork(TrainingModel):
    '''
    basic neural network
    '''

    def __init__(self, layers, X, y, learning_rate = 1, weight_decay = 0, activation_fn=TrainingModel.sigmoid, number_of_epochs=10, plot_cost_graph=False):


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
            layer_activations.append(z)
            a = self.activation_fn(z)
            layer_output.append(a)

        y_approx = a

        return layer_activations, layer_output, y_approx


    def cost(self):
        '''
        compute to cost
        '''

        no_exs = self.X.shape[0]

        reg_term = 0
        for weight in self.weights:
            reg_term += np.sum(np.power(weight, 2))

        _, _, y_approx = self.foward_feed(self.X)
        cost = 1/no_exs * np.sum(1/2 * np.power(y_approx - self.y, 2)) + self.weight_decay / 2 * reg_term

        return cost


    # TODO: support batch back prop
    def back_prop(self, x, y):
        '''
        perform back propigation to updated the weights
        '''

        # the list of deltas
        delta = []
        self.grad = []
        self.bias_grad = []


        layer_activations, layer_output, _ = self.foward_feed(x)

        a = layer_output[-1]
        z = layer_activations[-1]

        # calculate the gradient of the output layer
        delta.insert(0, -(y - a) * self.activation_fn_prime(z))

        a = layer_output[len(layer_output)-2]


        # the combination or weights and activation values does not readily
        # match up.  SO, i am going to manually force the arrays to the correct
        # size and zip em together!  I am also going to reverse the arrays,
        # since that's how we will need them
        weights = reversed(self.weights)

        # insert padding so arrays match in length
        layer_activations.insert(0, np.array([0]))

        layer_act = reversed(layer_activations[:-1])

        layer_out = reversed(layer_output[:-1])

        for weight, z, a in zip(weights, layer_act, layer_out):
            self.grad.insert(0, np.matmul(delta[0], a).T)
            self.bias_grad.insert(0, delta[0].T)

            # TODO: this is terrible.  rewrite...

            if len(delta) == len(self.weights):
                break

            delta.insert(0, np.matmul(weight, delta[0]) * self.activation_fn_prime(z).T)


    def train_model(self):
        no_exs = self.X.shape[0]

        for epoch in range(self.number_of_epochs):


            for i in range(self.X.shape[0]):
                self.back_prop(self.X[[i],:], self.y[[i],:])

                for i, _ in enumerate(self.weights):

                    self.weights[i] = self.weights[i] - self.learning_rate * (1/no_exs * self.grad[i] + self.weight_decay * self.weights[i])
                    self.biases[i] = self.biases[i] - 1/no_exs * self.learning_rate * self.bias_grad[i]

            if self.plot_cost_graph:
                if epoch % np.round(self.number_of_epochs/100) == 0:
                    plt.plot(epoch, self.cost(), 'ro')

            if epoch % np.round(self.number_of_epochs/10) == 0:
                print(self.cost())


        if self.plot_cost_graph:
            plt.show()




def main():
    # dummy example.  Will eventually move to neural_network_test.py
    #nn = NeuralNetwork(layers=np.array([15, 10, 1]), X=np.array([[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], [1,2.3,3.5,4,5,6,7,8,9,19,11,12,13,14,15], [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]]).reshape(3, 15), y=np.array([-1,1,1]).reshape(3,1), learning_rate=.001)
    X = np.array([[1,1], [1,3], [4,4]]).reshape(3,2)
    y = np.array([0, 10, 10]).reshape(3,1)
    learning_rate = .03
    layers = np.array([2, 3, 1])
    weight_decay = .0#001
    number_of_epochs = 500

    nn = NeuralNetwork(layers=layers, X=X, y=y, learning_rate=learning_rate, weight_decay=weight_decay, activation_fn=TrainingModel.rect_lin, number_of_epochs=number_of_epochs, plot_cost_graph=True)

    test = np.array([[4, 4]]).reshape(1,2)
    _,_,approx = nn.foward_feed(test)

    print(approx)


    nn.train_model()

    test = np.array([[4, 4]]).reshape(1,2)
    _,_,approx = nn.foward_feed(test)

    print(approx)


if __name__ == "__main__":
    main()
