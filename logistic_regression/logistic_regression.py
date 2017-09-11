import numpy as np
import matplotlib as pyplot
from ../training_model/training_model import TrainingModel

class LogisticRegression(TrainingModel):
    def __init__():
        # TODO: initialize based on superclass
        super.__init__()


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

