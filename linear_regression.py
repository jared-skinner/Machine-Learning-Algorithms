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
        self.cost = - np.sum(1/2 * np.power(self.y - np.matmul(self.X, self.weights), 2))


    def calculate_grad(self):
        self.grad = -(self.y - np.matmul(self.X, self.weights)) * self.X


    def train_model(self):
        for epoch in range(self.number_of_epochs):

            # calculate cost
            self.calculate_cost()

            # calculate grad
            self.calculate_grad()

            # adjust weights
            self.weights -= self.grad

            print("Cost: %f" % self.cost)

        # return weights
        return self.weights


    def train_model(self):
        # training weights

        for epoch in range(self.number_of_epochs):
            # calculate cost
            self.calculate_cost()
                
            # calculate gradient
            self.calculate_grad()

            # adjust weights
            self.weights -= self.grad
            
            if (epoch + 1) % 6000 == 0 or epoch == 0:
                print("Cost = %f" % self.cost)

                # TODO: this logic is broken!
                if 'DISPLAY' in environ.keys():
                    X_plot_vals = np.arange(0, 10, .001)

                    X_plot_powers = []
                    for i in range(0,7):
                        x_pow = np.power(X_plot_vals, i)
                        X_plot_powers.append(x_pow)

                    y_plot_vals = np.matmul(self.weights, X_plot_powers)
                    plt.plot(X, y, 'ro')
                    plt.plot(X_plot_vals, y_plot_vals)
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

    # we know from the dummy data that a 6th degree polynomial should model this
    # well.  With that in mind, we will create a feature for each power of X: 0
    # through 7.
    X_features = []
    for i in range(0, 7):
        x_pow = np.power(X, i)
        X_features.append(x_pow)

    # here is our new features vector!
    X_features = np.array(X_features)

    if 'DISPLAY' in environ.keys():
        # plot X and y to get an idea what we are looking at.
        plt.plot(X, y, 'ro')
        plt.show()

    # learning rate and number of epochs.  TODO: figure out why the learning rate
    # has to be so low to prevent divergence
    learning_rate = .000000000000000000001
    number_of_epochs = 100

    linear_model = LinearRegression(X, y, learning_rate, number_of_epochs)

    linear_model.train_model()

if __name__ == "__main__":
    main()


