import numpy as np
import matplotlib as pyplot
from training_model import TrainingModel


# TODO: add in bias terms
class NeuralNetwork(TrainingModel):
    '''
    basic neural network
    '''

    def __init__(self, layers, X, y):

        # layers should be a list (or np array) of sizes, where each index
        # represents a layer from left to right, and the number is the number of
        # nodes
        self.layers = layers
        self.number_of_hidden_layers = len(self.layers) - 2
        self.X = X
        self.y = y

        self.weights = []
        # calculate weights, place these in a list of np arrays
        for layer, next_layer in zip(layers, layers[1:]):
            self.weights.append(np.random.rand(layer, next_layer))

        # bias value for everything except the output layer
        self.bias = np.ones(len(self.layers) - 1)


        #super(NeuralNetwork, self).__init__()


    def foward_feed(self):
        # a is the input values of the current layer.  We will start with X
        a = self.X

        for weight in self.weights:
            z = np.matmul(z, weight)
            a = self.tanh(a)

        return a


    def back_prop(self):
        pass


def main():
    # dummy example.  Will eventually move to neural_network_test.py
    nn = NeuralNetwork(np.array([15, 5, 5, 1]), np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]), 16)

    #print(nn.weights)

    print(nn.foward_feed())


if __name__ == "__main__":
    main()
