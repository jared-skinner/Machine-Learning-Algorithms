import unittest
import numpy as np
from neural_network import NeuralNetwork
from training_model import TrainingModel


# TODO: fill these out and learn how to do TDD!!!
class TestHappy(unittest.TestCase):

    def test_foward_feed():
        pass

    def test_cost():
        pass

    def test_back_prop():
        pass


def main():
    # dummy example
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
