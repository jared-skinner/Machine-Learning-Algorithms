import numpy as np
import matplotlib.pyplot as plt
from training_model import TrainingModel

# environ to see if there is a display defined for showing off plots
from os import environ

class LinearRegression(TrainingModel):
    def __init__(self, X, y, learning_rate, number_of_epochs):
        super(LinearRegression, self).__init__(X, y, learning_rate, number_of_epochs)


    def calculate_cost(self):
        '''
        compute the cost of the logistic algorithm with weights <W>
        '''
        self.cost = np.sum(1/2 * np.power(self.y - np.matmul(self.X, np.transpose(self.weights)), 2))


    def calculate_grad(self):
        self.grad = -np.matmul(np.transpose(self.y - np.matmul(self.X, np.transpose(self.weights))), self.X)


    def train_model(self):
        # training weights

        for epoch in range(self.number_of_epochs):

            # calculate gradient
            self.calculate_grad()

            # adjust weights
            self.weights = self.weights - self.learning_rate * self.grad
            
            if (epoch + 1) % 6000 == 0 or epoch == 0:

                # calculate cost
                self.calculate_cost()
                print("Cost = %f" % self.cost)

                if 'DISPLAY' in environ.keys():
                    y_predict = np.matmul(self.X, np.transpose(self.weights))

                    plt.plot(self.X[:,1], self.y, 'ro')
                    plt.plot(self.X[:,1], y_predict, 'bo')
                    plt.show()
            
        return self.weights


    def test_model(self):
        pass


def main():
    # use this for testing the model

    # define some dummy data
    X = np.sort(np.random.rand(30)) * 10
    noise = (np.sort(np.random.rand(30)) - .5) * 30
    y = np.multiply(- .002 * np.power(X, 6) + .3 * np.power(X, 5) + .002 * np.power(X, 4) + .00000000000033 * X + 10, noise)
    y  = y.reshape(30,1)

    # we know from the dummy data that a 6th degree polynomial should model this
    # well.  With that in mind, we will create a feature for each power of X: 0
    # through 7.
    X_features = []
    for i in range(0, 7):
        x_pow = np.power(X, i)
        X_features.append(x_pow)

    # here is our new features vector!
    X_features = np.transpose(np.array(X_features))

    if 'DISPLAY' in environ.keys():
        # plot X and y to get an idea what we are looking at.
        plt.plot(X, y, 'ro')
        plt.show()

    # learning rate and number of epochs.  TODO: figure out why the learning rate
    # has to be so low to prevent divergence
    learning_rate = .00000000000001
    number_of_epochs = 100000

    linear_model = LinearRegression(X_features, y, learning_rate, number_of_epochs)

    linear_model.train_model()

if __name__ == "__main__":
    main()

