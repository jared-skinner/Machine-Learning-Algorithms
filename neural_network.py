import numpy as np
import matplotlib as pyplot
from training_model import TrainingModel


# TODO: add in bias terms
class NeuralNetwork(TrainingModel):
    '''
    basic neural network
    '''

    def __init__(self, layers, X, y, learning_rate = 1, lam = 0):

        # layers should be a list (or np array) of sizes, where each index
        # represents a layer from left to right, and the number is the number of
        # nodes
        self.layers = layers
        self.number_of_hidden_layers = len(self.layers) - 2

        assert X.shape[0] == y.shape[0]

        # regularization parameter.  real number
        self.lam = lam

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

        self.grad = []

        self.learning_rate = learning_rate


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
            z = np.matmul(a, weight)# + bias
            layer_activations.append(z)
            a = self.tanh(z)
            layer_output.append(a)

        y_approx = a

        return layer_activations, layer_output, y_approx



    # TODO: add in regulization term
    def cost(self):
        '''
        compute to cost
        '''

        no_exs = self.X.shape[0]

        _, _, y_approx = self.foward_feed(self.X)
        cost = 1/no_exs * np.sum(1/2 * np.power(y_approx - self.y, 2))

        return cost


    def back_prop(self, x, y):
        '''
        perform back propigation to updated the weights
        '''

        # the list of deltas
        delta = []
        self.grad = []


        layer_activations, layer_output, _ = self.foward_feed(x)

        a = layer_output[len(layer_output)-1]
        z = layer_activations[len(layer_activations)-1]

        # calculate the gradient of the output layer
        delta.insert(0, -(y - a) * self.tanh_prime(z))

        a = layer_output[len(layer_output)-2]


        # the combination or weights and activation values does not readily
        # match up.  SO, i am going to manually force the arrays to the correct
        # size and zip em together!  I am also going to reverse the arrays,
        # since that's how we will need them

        weights = reversed(self.weights)

        layer_act = reversed(layer_activations)

        layer_out = reversed(layer_output[:-1])

        for weight, z, a in zip(weights, layer_act, layer_out):
            self.grad.insert(0, np.transpose(np.matmul(delta[0], a)))

            # TODO: this is terrible.  rewrite...
            if len(delta) == len(self.weights):
                break

            delta.insert(0, np.matmul(weight, delta[0]) * np.transpose(self.tanh_prime(z)))


def main():
    # dummy example.  Will eventually move to neural_network_test.py
    nn = NeuralNetwork(layers=np.array([15, 5, 5, 1]), X=np.array([[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], [1,2.3,3.5,4,5,6,7,8,9,19,11,12,13,14,15], [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]]).reshape(3, 15), y=np.array([1,2,3]).reshape(3,1), learning_rate=.001)

    number_of_epochs = 10

    for epoch in range(number_of_epochs):
        print(nn.cost())

        for i in range(nn.X.shape[0]):
            nn.back_prop(nn.X[[i],:], nn.y[[i],:])

        for i, _ in enumerate(nn.weights):
            nn.weights[i] = nn.weights[i] - nn.learning_rate * nn.grad[i]


if __name__ == "__main__":
    main()
