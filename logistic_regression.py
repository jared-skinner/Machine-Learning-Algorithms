import numpy as np
import matplotlib as pyplot
from training_model import TrainingModel

class LogisticRegression(TrainingModel):
    def __init__(self, X, y, learning_rate, number_of_epochs):
        super(LogisticRegression, self).__init__(X, y, learning_rate, number_of_epochs)


    def calculate_prediction(self):

        #          X                Weights
        #
        # examples X features * features X 1   

        z = np.matmul(self.X, self.weights)
        return self.sigmoid(z)


    def calculate_cost(self):
        '''
        compute the cost of the logistic algorithm with weights <W>
        '''
        z = np.matmul(self.X, self.weights)
        self.cost = - np.sum(self.y * np.log(self.sigmoid(z)) + (1 - self.y) * np.log(1 - self.sigmoid(z)))


    def calculate_grad(self):
        y_predict = self.calculate_prediction()

        self.grad = np.matmul(np.transpose(self.X), y_predict - self.y)


    def train_model(self):
        for epoch in range(self.number_of_epochs):

            # calculate grad
            self.calculate_grad()

            # adjust weights
            self.weights = self.weights - self.learning_rate * self.grad

            # calculate cost
            self.calculate_cost()

            print("Cost: %f" % self.cost)

        # return weights
        return self.weights


    def test_model(self):
        pass


def main():
    # basic test data.  As this becomes more robust it will be moved to its own
    # file
    X = np.array([1, 1.2, 1.3, .9, 7, 8.1, 7.5, 7.7]).reshape(8, 1)
    y = np.array([0,   0,   0,  0, 1,   1,   1,   1]).reshape(8, 1)
    learning_rate = .1
    number_of_epochs = 200
 
    logistic = LogisticRegression(X, y, learning_rate, number_of_epochs)

    # randomize data
    logistic.shuffle_data()

    # split data into train and test
    logistic.split_data(.7, .3, 0)

    weights = logistic.train_model()

    logistic.test_model()


if __name__ == "__main__":
    main()

