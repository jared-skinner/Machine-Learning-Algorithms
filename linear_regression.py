import numpy as np
import matplotlib.pyplot as plt
from os import environ

def train_model(X, y, number_of_epochs, learning_rate):
    # training weights
    W = (np.random.rand(X.shape[0]) - .5) * 10

    for epoch in range(number_of_epochs):
        # create predictions
        y_approx = np.matmul(W, X)

        # calculate cost
        cost = np.sum(1/2 * np.power(y - y_approx,2))
            
        # calculate gradient
        grad = np.sum(-(y - y_approx) * X, axis=1)

        # move weights
        W = W - learning_rate * grad
        
        if (epoch + 1) % 6000 == 0 or epoch == 0:
            print("Cost = %f" % cost)

            if 'DISPLAY' in environ.keys():
                X_plot_vals = np.arange(0, 10, .001)

                X_plot_powers = []
                for i in range(0,7):
                    x_pow = np.power(X_plot_vals, i)
                    X_plot_powers.append(x_pow)

                y_plot_vals = np.matmul(W, X_plot_powers)
                plt.plot(X, y, 'ro')
                plt.plot(X_plot_vals, y_plot_vals)
                plt.show()
        
    print("==========================================================")
    print("================== TRAINING COMPLETE =====================")
    print("==========================================================")
        
    print("Weights: ")
    print(W)
    print("Cost: %f" % cost)  

    return W


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
    learning_rate = .0000000000001
    number_of_epochs = 100000

    weights = train_model(X_features, y, number_of_epochs, learning_rate)

if __name__ == "__main__":
    main()


