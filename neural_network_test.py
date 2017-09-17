import unittest
import numpy as np
from neural_network import NeuralNetwork
from training_model import TrainingModel

import scipy.misc
import matplotlib.pyplot as plt
import os
import pickle






# TODO: fill these out and learn how to do TDD!!!
class TestHappy(unittest.TestCase):

    def test_foward_feed():
        pass

    def test_cost():
        pass

    def test_back_prop():
        pass


def main():

    image_paths = os.listdir("mnist/Images/train/")

    images = []
    y_vals = []



    if os.path.isfile("mnist_train.pickle"):
        with open("mnist_train.pickle", 'rb') as mnist_pickle:
            pickled_data = pickle.load(mnist_pickle)
            y_vals, images = pickled_data[0], pickled_data[1]
    else:
        with open("mnist/train.csv") as train_results:
            for line in train_results:
                image_path, y_val = line.split(",")

                if image_path == "filename":
                    continue

                # read image from training data
                image = scipy.misc.imread(os.path.join("mnist/Images/train/", image_path), flatten=True)

                # read in corresponding value from csv file

                image = np.ndarray.flatten(image)

                images.append(image)

                y_val = int(y_val)
                y_vals.append(TrainingModel.digit_to_one_hot(10, y_val))

        y_vals = np.array(y_vals)
        images = np.array(images)

        pickled_data = [y_vals, images]


        with open("mnist_train.pickle", 'wb') as mnist_pickle:
            pickle.dump(pickled_data, mnist_pickle)

    # dummy example
    X = images
    y = y_vals

    learning_rate = .1
    layers = np.array([784, 15, 10])
    weight_decay = .00001
    number_of_epochs = 10
    activation_fn = TrainingModel.sigmoid

    nn = NeuralNetwork(layers=layers, X=X, y=y, learning_rate=learning_rate, weight_decay=weight_decay, activation_fn=activation_fn, number_of_epochs=number_of_epochs, plot_cost_graph=True)

    nn.train_model()

    #print("approx: %d" % TrainingModel.one_hot_to_digit(approx))
    #print("actual: %d" % TrainingModel.one_hot_to_digit(y_vals[1]))


def dumb_example():
    X = np.array([[2,1], [1,3], [4,4]]).reshape(3,2)
    y = np.array([0, 1, 1]).reshape(3,1)
    learning_rate = .1
    layers = np.array([2, 1, 1])
    weight_decay = 0
    number_of_epochs = 10000

    nn = NeuralNetwork(layers=layers, X=X, y=y, learning_rate=learning_rate, weight_decay=weight_decay, activation_fn=TrainingModel.sigmoid, number_of_epochs=number_of_epochs, plot_cost_graph=True)

    test = np.array([[4, 4]]).reshape(1,2)
    _,_,approx = nn.foward_feed(test)
#
    print(approx)

    nn.train_model()

    _,_,approx = nn.foward_feed(test)

    print(approx)

if __name__ == "__main__":
    main()
