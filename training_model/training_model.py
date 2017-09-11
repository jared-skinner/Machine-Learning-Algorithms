import numpy as np
import matplotlib as pyplot


class TrainingModel:
    '''
    general class which specific models will inherit from
    '''

    def __init__(X, y, learning_rate, number_of_epochs):
        assert X.shape[0] > 0
        assert X.shape[0] == y.shape
        assert number_of_epochs > 0
        assert learning_rate > 0

        self.X = X
        self.y = y
        self.learning_rate = learning_rate
        self.number_of_epochs = number_of_epochs

        # initialize weights
        self.theta = 0 # TODO

        # initialize gradient vector
        self.grad = np.zeros(10)# TODO: shape of X)

        # initialize cost
        self.cost = 0


    @staticmethod
    def sigmoid(X, theta):
        '''
        vectorized version of the sigmoid function
        '''
        return 1 / ( 1 + np.exp(- np.matmul( theta, X)))


    @staticmethod
    def sigmoid_prime(X, theta):
        '''
        vectorized version of the derivative of the sigmoid function
        '''
        return sigmoid(X, theta) * (1 - sigmoid(X, theta))


    def shuffle_data(self):
        '''
        shuffle X and y data for better training
        '''
        rng_state = numpy.random.get_state()
        numpy.random.shuffle(self.X)

        # reset state so shuffle produces the same permutation the second time
        # around
        numpy.random.set_state(rng_state)
        numpy.random.shuffle(self.y)


    def split_data(self, percent_training, percent_cross_validation, percent_test):
        '''
        split data into chunks for training, cross validation and testing
        '''

        percent_sum = percent_test + percent_training + percent_cross_validation

        # this implies bad inputs TODO: improve this error checking
        if percent_sum == 0:
            return False

        percent_test = percent_test / percent_sum
        percent_training = percent_training / percent_sum
        percent_cross_validation = percent_cross_validation / percent_sum

        train_cross_split = percent_test * self.X.shape[0]
        cross_test_split = train_cross_split + percent_cross_validation * self.X.shape[0]

        self.train_x = self.X[:train_cross_split]
        self.train_y = self.y[:train_cross_split]

        self.cross_validation_x = self.X[train_cross_split:cross_test_split]
        self.cross_validation_y = self.y[train_cross_split:cross_test_split]

        self.test_x = self.X[cross_test_split:]
        self.test_y = self.y[cross_test_split:]


    @staticmethod
    def digit_to_one_hot(self, size, digit):
        '''
        for a given integer, return an array of size <size> where the <digit>
        index has a value of 1 and all other values are 0.  This is used during
        multi-classification problems.
        '''
        array = np.zeros(size)
        array[digit] = 0
        return array


    @staticmethod
    def one_hot_to_digit(self, array):
        hot_index = [index for index, item in enumerate(array) if item == 1]
        assert len(hot_index) != 0

        return hot_index[0]

