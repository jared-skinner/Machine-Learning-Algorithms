import numpy as np
import matplotlib as pyplot


class TrainingModel:
    '''
    general class which specific models will inherit from
    '''

    def __init__(self, X, y, learning_rate, number_of_epochs):
        assert X.shape[0] > 0
        assert X.shape[0] == y.shape[0]
        assert number_of_epochs > 0
        assert learning_rate > 0

        self.X = X
        self.y = y
        self.learning_rate = learning_rate
        self.number_of_epochs = number_of_epochs

        self.weights = np.zeros((1, self.X.shape[1]))

        # initialize cost
        self.cost = 0


    @staticmethod
    def sigmoid(X):
        '''
        vectorized version of the sigmoid function
        '''
        return np.power(( 1 + np.exp(-X)), -1)


    @staticmethod
    def sigmoid_prime(X):
        '''
        vectorized version of the derivative of the sigmoid function
        '''
        return TrainingModel.sigmoid(X) * (1 - TrainingModel.sigmoid(X))


    @staticmethod
    def tanh(X):
        return (np.exp(X) - np.exp(-X))/(np.exp(X) + np.exp(-X))


    @staticmethod
    def tanh_prime(X):
        return 1 - np.power(TrainingModel.tanh(X), 2)


    @staticmethod
    def rect_lin(X):
        return np.maximum(X, 0)


    @staticmethod
    def rect_lin_prime(X):
        return X > 0


    def shuffle_data(self):
        '''
        shuffle X and y data for better training
        '''
        rng_state = np.random.get_state()
        np.random.shuffle(self.X)

        # reset state so shuffle produces the same permutation the second time
        # around
        np.random.set_state(rng_state)
        np.random.shuffle(self.y)


    def split_data(self, percent_training = 0, percent_cross_validation = 0, percent_test = 0):
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

        train_cross_split = int(percent_test * self.X.shape[0])
        cross_test_split = int(train_cross_split + percent_cross_validation * self.X.shape[0])

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

