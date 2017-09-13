import numpy as np
import matplotlib as pyplot
from training_model import TrainingModel


class NeuralNetwork(TrainigModel):
    '''
    basic neural network
    '''

    def __init__(self, layers):
        self.layers = layers
        self.number_of_hidden_layers = len(self.layers  - 2)

        super(NeuralNetwork, self).__init__()


    def foward_feed(self):
        pass


    def back_prop(self):
        pass

