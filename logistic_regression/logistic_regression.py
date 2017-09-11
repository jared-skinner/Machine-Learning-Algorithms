import numpy as np
import matplotlib as pyplot
from ../training_model/training_model import TrainingModel

class LogisticRegression(TrainingModel):
    def __init__(X, y, learning_rate, number_of_epochs):
        # TODO: initialize based on superclass
        super(LogisticRegression, self).__init__(X, y, learning_rate, number_of_epochs)


    def calculate_cost(self, X_vals, Y_vals, W):
        '''
        compute the cost of the logistic algorithm with weights <W>
        '''
        self.cost = - np.sum(self.y * np.log(self.sigmoid(X, self.weights)) + (1 - y) * np.log(1 - self.sigmoid(X, self.weights)))


    def calculate_grad(self):
        self.grad = np.sum(self.X * (self.sigmoid(self.X) - self.y[i]))


    def train_model(self):
        for epoch in range(number_of_epochs):

            # calculate cost
            self.calculate_cost()

            # calculate grad
            self.calculate_grad()

            # adjust weights
            self.weights -= self.grad

        # return weights
        return self.weights


    def test_model(self):
        pass


def main():
    logistic = LogisticRegression()

    # define dummy data features x and y

    # randomize data
    logistic.shuffle_data()

    # split data into train and test
    logistic.split_data(.7, .3, 0)

    weights = logistic.train_model()

    logistic.test_model()


if __name__ == "__main__":
    main()

