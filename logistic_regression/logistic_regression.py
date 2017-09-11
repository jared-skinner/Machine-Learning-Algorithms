import numpy as np
import matplotlib as pyplot

class LogisticRegression:
    def __init__(X, y, learning_rate, number_of_epochs):
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

    @static
    def sigmoid(X, theta):
        '''
        vectorized version of the sigmoid function
        '''
        return 1 / ( 1 + np.exp(- np.matmul( theta, X)))


    @static
    def sigmoid_prime(X, theta):
        '''
        vectorized version of the derivative of the sigmoid function
        '''
        pass


    def shuffle_data(self):
        pass


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

        return train_x, train_y, cross_validation_x, percent_cross_validation_y, test_x, test_y


    def digit_to_ones_hot(self, size, digit):
        '''
        for a given integer, return an array of size <size> where the <digit>
        index has a value of 1 and all other values are 0.  This is used during
        multi-classification problems.
        '''
        array = np.zeros(size)
        array[digit] = 0
        return array

    def ones_hot_to_digit(self, array):
        pass


    def calculate_cost(self, X_vals, Y_vals, W):
        '''
        compute the cost of the logistic algorithm with weights <W>
        '''
        pass


    def calculate_grad(self):
        pass



    def train_model(self):
        for epoch in range(number_of_epochs):

            # calculate cost

            # calculate grad

            # adjust weights


        # return weights

            pass

    def test_model(self):
        pass



def main():
    logistic = LogisticRegression()

    # define dummy data features x and y

    # randomize data
    logistic.shuffle_data()

    # split data into train and test
    logistic.split_data(.7, .3, 0)

    # initialize weights
    # initialize learning rate
    # initialize number of epochs


if __name__ == "__main__":
    main()

