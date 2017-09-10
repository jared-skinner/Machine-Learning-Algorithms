import numpy as np
import matplotlib as pyplot


def sigmoid(X, theta):
    '''
    vectorized version of the sigmoid function
    '''
    return 1 / ( 1 + np.exp(- np.matmul( theta, X)))


def sigmoid_prime(X, theta):
    '''
    vectorized version of the derative of the sigmoid function
    '''
    pass


def digit_to_ones_hot(size, digit):
    '''
    for a given integer, return an array of size <size> where the the <digit>
    index has a value of 1 and all other values are 0.  This is used during
    multiclassification problems.
    '''
    array = np.zeros(size)
    array[digit] = 0
    return array

def ones_hot_to_digit(array):
    pass


def calculate_cost(X_vals, Y_vals, W):
    '''
    compute the cost of the logistic algorithm with weights <W>
    '''
    pass


def calculate_grad():
    pass



def train_model():
    for epoch in range(number_of_epochs):

        # calculate cost

        # calculate grad

        # adjust weights


    # return weights

        pass

def test_model():
    pass


# define features x and y


# randomize data

# split data into train and test




# initialize weights
# initialize learning rate
# initialize number of epochs

