import unittest
import numpy as np
from neural_network import NeuralNetwork
from training_model import TrainingModel

from scipy.misc import imread
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


def mnist_test():
    images = []
    y_vals = []

    test_image_paths = os.listdir("mnist/Images/test")

    # if mnist data has already been processed, then load the pickle, otherwise
    # process the data
    if os.path.isfile("mnist/mnist_test.pickle"):
        with open("mnist/mnist_test.pickle", 'rb') as mnist_pickle:
            pickled_data = pickle.load(mnist_pickle)
            y_vals, images = pickled_data[0], pickled_data[1]
    else:
        #with open("mnist/train.csv") as train_results:
        #    for line in train_results:
        #        image_path, y_val = line.split(",")
#
#                # read image from training data
#                if image_path == "filename" or not os.path.exists(os.path.join("mnist/Images/test", image_path)):
#                    continue
#
#                image = imread(os.path.join("mnist/Images/test", image_path), flatten=True)
#
#                image = np.ndarray.flatten(image)
#
#                images.append(image)
#
#                y_val = int(y_val)
#                y_vals.append(TrainingModel.digit_to_one_hot(10, y_val))

        for path in test_image_paths:
            image = imread(os.path.join("mnist/Images/test", path), flatten=True)

            image = np.ndarray.flatten(image)

            images.append(image)

        images = np.array(images)

    # regularize so we don't saturate the model
    X = images/255
    y = []

    learning_rate =  3
    layers = np.array([784, 30, 10])
    weight_decay = 0#.00001
    number_of_epochs = 1000
    activation_fn = TrainingModel.sigmoid
    number_of_batches = 1

    nn = NeuralNetwork(layers=layers, X=X, y=y, learning_rate=learning_rate, weight_decay=weight_decay, activation_fn=activation_fn, number_of_epochs=number_of_epochs, plot_cost_graph=True, number_of_batches=number_of_batches)

    with open("mnist/weights.pickle", 'rb') as mnist_pickle:
        weights = pickle.load(mnist_pickle)

    nn.weights = weights

    _,_,y_approx = nn.foward_feed(X)

    total = 0
    total_right = 0
    for x_val, y_app in zip(X, y_approx):

        y_digit_approx = TrainingModel.one_hot_to_digit(y_app)

        print("approx: %d" % y_digit_approx)
        print("\n")

        # plot the image
        plt.imshow(x_val.reshape(28, 28))
        plt.show()


def mnist_train():
    images = []
    y_vals = []

    # if mnist data has already been processed, then load the pickle, otherwise
    # process the data
    if os.path.isfile("mnist/mnist_train.pickle"):
        with open("mnist/mnist_train.pickle", 'rb') as mnist_pickle:
            pickled_data = pickle.load(mnist_pickle)
            y_vals, images = pickled_data[0], pickled_data[1]
    else:
        with open("mnist/train.csv") as train_results:
            for line in train_results:
                image_path, y_val = line.split(",")

                if image_path == "filename":
                    continue

                # read image from training data
                image = imread(os.path.join("mnist/Images/train/", image_path), flatten=True)

                image = np.ndarray.flatten(image)

                images.append(image)

                y_val = int(y_val)
                y_vals.append(TrainingModel.digit_to_one_hot(10, y_val))

        y_vals = np.array(y_vals)
        images = np.array(images)

        pickled_data = [y_vals, images]


        with open("mnist/mnist_train.pickle", 'wb') as mnist_pickle:
            pickle.dump(pickled_data, mnist_pickle)

    # regularize so we don't saturate the model
    X = images/255
    y = y_vals

    learning_rate =  3
    layers = np.array([784, 30, 10])
    weight_decay = 0#.00001
    number_of_epochs = 1000
    activation_fn = TrainingModel.sigmoid
    number_of_batches = 1

    nn = NeuralNetwork(layers=layers, X=X, y=y, learning_rate=learning_rate, weight_decay=weight_decay, activation_fn=activation_fn, number_of_epochs=number_of_epochs, plot_cost_graph=True, number_of_batches=number_of_batches)

    nn.train_model()

    with open("mnist/weights.pickle", 'wb') as mnist_weights_pickle:
        pickle.dump(nn.weights, mnist_weights_pickle)

    test = np.zeros(784)
    for i in range(X.shape[0]):

        test = X[i]
        actual = y[i]

        _,_,approx = nn.foward_feed(test)

        print("approx: %d" % TrainingModel.one_hot_to_digit(approx))
        print("actual: %d" % TrainingModel.one_hot_to_digit(actual))
        print("\n")

        # plot the image
        plt.imshow(images[i].reshape(28, 28))
        plt.show()

        if i == 100:
            break


def dumb_example():
    X = np.array([[2,1], [1,3], [4,4]]).reshape(3,2)
    y = np.array([0, 1, 1]).reshape(3,1)
    learning_rate = .1
    layers = np.array([2, 3, 1])
    weight_decay = 0
    number_of_epochs = 10000

    nn = NeuralNetwork(layers=layers, X=X, y=y, learning_rate=learning_rate, weight_decay=weight_decay, activation_fn=TrainingModel.sigmoid, number_of_epochs=number_of_epochs, plot_cost_graph=True)

    test = np.array([[4, 4]]).reshape(1,2)
    _,_,approx = nn.foward_feed(test)
#
    nn.train_model()

    _,_,approx = nn.foward_feed(test)

    print(approx)

if __name__ == "__main__":
    dumb_example()
